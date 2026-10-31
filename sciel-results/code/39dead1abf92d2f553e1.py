"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_2block(a, b, m, N):
    """Compute max overlap for 2-block construction with heights a,b and split at m."""
    ov1 = a * (1.0 - a)
    ov_trans = a * (1.0 - b)
    ov2 = b * (1.0 - b)
    ov_end = b
    return max(ov1, ov_trans, ov2, ov_end)


def _soft_max(vals, beta=500.0):
    """Smooth approximation to max via log-sum-exp."""
    m = np.max(vals)
    if m < 0:
        return m
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta


def _soft_overlap_2block(a, b, m, N, beta=500.0):
    """Smooth max overlap for 2-block construction."""
    ov1 = a * (1.0 - a)
    ov_trans = a * (1.0 - b)
    ov2 = b * (1.0 - b)
    ov_end = b
    vals = np.array([ov1, ov_trans, ov2, ov_end])
    return _soft_max(vals, beta)


def _objective_smooth(params, beta=500.0):
    """Smooth objective: minimize soft-max overlap, params = [a, m_continuous]."""
    a, m_c = params
    if m_c <= 0.5 or m_c >= N - 0.5:
        return 1e10
    b = (N / 2.0 - m_c * a) / (N - m_c)
    if b < -0.01 or b > 1.01:
        return 1e10
    b = np.clip(b, 0.0, 1.0)
    return _soft_overlap_2block(a, b, m_c, N, beta)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N / 2.0

    best_score = _overlap_for_2block(A0, B0, m0, N)
    best_params = (A0, B0, m0)

    # Phase 1: Coarse search with Nelder-Mead on smooth objective
    beta_coarse = 100.0
    best_val = 1e10
    best_x = None

    for a_init in [A0, 0.3, 0.4, 0.5, 0.35, 0.45]:
        for m_init in [N/2.0, N*0.4, N*0.55, N*0.45, N*0.52]:
            x0 = np.array([a_init, m_init])
            try:
                res = minimize(lambda x: _objective_smooth(x, beta_coarse), x0,
                               method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    if best_x is not None:
        a_opt = float(best_x[0])
        m_cont = float(best_x[1])
        b_opt = (N / 2.0 - m_cont * a_opt) / (N - m_cont)
        b_opt = np.clip(b_opt, 0.0, 1.0)
        m_rounded = int(round(m_cont))
        m_rounded = max(1, min(N - 1, m_rounded))
        b_int = (N / 2.0 - m_rounded * a_opt) / (N - m_rounded)
        b_int = np.clip(b_int, 0.0, 1.0)
        score_int = _overlap_for_2block(a_opt, b_int, m_rounded, N)
        if score_int < best_score:
            best_score = score_int
            best_params = (a_opt, b_int, m_rounded)

    # Phase 2: Per-m optimization with smooth objective
    beta_fine = 500.0
    for m_try in range(max(1, int(N * 0.3)), min(N - 1, int(N * 0.7) + 1)):
        if time.time() > deadline:
            break

        def obj_a_smooth(a_val, m=m_try, beta=beta_fine):
            b_val = (N / 2.0 - m * a_val) / (N - m)
            if b_val < 0 or b_val > 1:
                return 1e10
            return _soft_overlap_2block(a_val, b_val, m, N, beta)

        a_grid = np.linspace(0.1, 0.9, 50)
        best_a_local = a_grid[0]
        best_val_local = 1e10
        for a_g in a_grid:
            v = obj_a_smooth(a_g)
            if v < best_val_local:
                best_val_local = v
                best_a_local = a_g

        try:
            res = minimize(obj_a_smooth, np.array([best_a_local]),
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16})
            a_ref = float(res.x[0])
            b_ref = (N / 2.0 - m_try * a_ref) / (N - m_try)
            b_ref = np.clip(b_ref, 0.0, 1.0)
            score_ref = _overlap_for_2block(a_ref, b_ref, m_try, N)
            if score_ref < best_score:
                best_score = score_ref
                best_params = (a_ref, b_ref, m_try)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        beta_high = 2000.0
        a_final, b_final, m_final = best_params
        m_cont_final = float(m_final)

        def polish_obj(x, beta=beta_high):
            a, m_c = x
            if m_c <= 0.5 or m_c >= N - 0.5:
                return 1e10
            b = (N / 2.0 - m_c * a) / (N - m_c)
            if b < -0.01 or b > 1.01:
                return 1e10
            b = np.clip(b, 0.0, 1.0)
            return _soft_overlap_2block(a, b, m_c, N, beta)

        x0 = np.array([a_final, m_cont_final])
        try:
            res = minimize(polish_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            a_polish = float(res.x[0])
            m_polish = float(res.x[1])
            b_polish = (N / 2.0 - m_polish * a_polish) / (N - m_polish)
            b_polish = np.clip(b_polish, 0.0, 1.0)
            m_rounded = int(round(m_polish))
            m_rounded = max(1, min(N - 1, m_rounded))
            b_int = (N / 2.0 - m_rounded * a_polish) / (N - m_rounded)
            b_int = np.clip(b_int, 0.0, 1.0)
            score_polish = _overlap_for_2block(a_polish, b_int, m_rounded, N)
            if score_polish < best_score:
                best_score = score_polish
                best_params = (a_polish, b_int, m_rounded)
        except Exception:
            pass

    a_final, b_final, m_final = best_params

    h = np.zeros(N)
    h[:m_final] = a_final
    h[m_final:] = b_final

    s = h.sum()
    diff = N / 2.0 - s
    if abs(diff) > 1e-15:
        if m_final > 0:
            h[m_final - 1] += diff
        else:
            h[-1] += diff

    h = np.clip(h, 0.0, 1.0)

    return h
