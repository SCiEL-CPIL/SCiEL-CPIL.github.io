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
    # Within block 1: a*(1-a)
    ov1 = a * (1.0 - a)
    # At transition (i=m-1): a*(1-b)
    ov_trans = a * (1.0 - b)
    # Within block 2: b*(1-b)
    ov2 = b * (1.0 - b)
    # At end (i=N-1): b*(1-0) = b
    ov_end = b
    return max(ov1, ov_trans, ov2, ov_end)


def _objective(params):
    """Objective: minimize max overlap, params = [a, b, m_continuous]."""
    a, b, m_c = params
    # Constraint: m*a + (N-m)*b = N/2, so b = (N/2 - m*a) / (N - m)
    # But we parameterize with a and m, derive b
    m = m_c
    if m <= 0 or m >= N:
        return 1e10
    b = (N / 2.0 - m * a) / (N - m)
    if b < -1e-10 or b > 1.0 + 1e-10:
        return 1e10
    b = np.clip(b, 0.0, 1.0)
    return _overlap_for_2block(a, b, m, N)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with theoretical 2-block solution
    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N / 2.0

    best_score = _overlap_for_2block(A0, B0, m0, N)
    best_params = (A0, B0, m0)

    # Optimize over continuous (a, m) where b is derived from constraint
    def obj_continuous(x):
        a, m_c = x
        if m_c <= 0.5 or m_c >= N - 0.5:
            return 1e10
        b = (N / 2.0 - m_c * a) / (N - m_c)
        if b < -0.01 or b > 1.01:
            return 1e10
        return _overlap_for_2block(a, b, m_c, N)

    # Try multiple initial points
    best_val = 1e10
    best_x = None

    for a_init in [A0, 0.3, 0.4, 0.5, 0.35, 0.45]:
        for m_init in [N/2.0, N*0.4, N*0.55, N*0.45, N*0.52]:
            x0 = np.array([a_init, m_init])
            try:
                res = minimize(obj_continuous, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
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

        # Recompute b with integer m to satisfy sum constraint exactly
        b_int = (N / 2.0 - m_rounded * a_opt) / (N - m_rounded)
        b_int = np.clip(b_int, 0.0, 1.0)

        # Check if this is better
        score_int = _overlap_for_2block(a_opt, b_int, m_rounded, N)
        if score_int < best_score:
            best_score = score_int
            best_params = (a_opt, b_int, m_rounded)

    # Also try optimizing with integer m directly for each candidate m
    for m_try in range(max(1, int(N * 0.3)), min(N - 1, int(N * 0.7) + 1)):
        if time.time() > deadline:
            break
        # For fixed m, optimize a (b derived)
        def obj_a(a_val):
            b_val = (N / 2.0 - m_try * a_val) / (N - m_try)
            if b_val < 0 or b_val > 1:
                return 1e10
            return _overlap_for_2block(a_val, b_val, m_try, N)

        # Grid search for initial guess
        a_grid = np.linspace(0.1, 0.9, 100)
        best_a_local = a_grid[0]
        best_val_local = 1e10
        for a_g in a_grid:
            v = obj_a(a_g)
            if v < best_val_local:
                best_val_local = v
                best_a_local = a_g

        # Refine with Nelder-Mead
        try:
            res = minimize(obj_a, np.array([best_a_local]), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15})
            a_ref = float(res.x[0])
            b_ref = (N / 2.0 - m_try * a_ref) / (N - m_try)
            b_ref = np.clip(b_ref, 0.0, 1.0)
            score_ref = _overlap_for_2block(a_ref, b_ref, m_try, N)
            if score_ref < best_score:
                best_score = score_ref
                best_params = (a_ref, b_ref, m_try)
        except Exception:
            pass

    a_final, b_final, m_final = best_params

    # Construct h
    h = np.zeros(N)
    h[:m_final] = a_final
    h[m_final:] = b_final

    # Fix sum to be exactly N/2
    s = h.sum()
    diff = N / 2.0 - s
    if abs(diff) > 1e-15:
        # Distribute correction across the last element of the smaller block
        if m_final > 0:
            h[m_final - 1] += diff
        else:
            h[-1] += diff

    # Guard: ensure values within [0,1]
    h = np.clip(h, 0.0, 1.0)

    return h
