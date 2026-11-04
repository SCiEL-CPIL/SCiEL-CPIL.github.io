import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 50
DEADLINE = 112.0  # safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=50.0):
    """||h||^2_2 / (||h||_1 * ||h*h||_inf) with soft-max for the peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    shifted = beta * (conv - max_val)
    exp_shifted = np.exp(shifted - np.max(shifted))
    log_sum_exp = np.max(shifted) + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=50.0):
    return -_ratio(h, beta)

def _dft_parameterize(z):
    H = np.fft.fft(z)
    return np.abs(H) ** 2

def _dft_inverse(h):
    H_mags = np.sqrt(np.maximum(h, 0.0) + 1e-30)
    return np.fft.ifft(H_mags.astype(complex))

def _local_optimize(h0, max_iter=200, tol=1e-9):
    h = np.maximum(h0, 0.0).copy()
    def objective(h_flat):
        return _ratio_neg(h_flat, beta=50.0)
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    h_opt = np.maximum(result.x, 0.0)
    return h_opt, _ratio(h_opt)

def _optimize_from_z(z0, max_iter=100, tol=1e-8):
    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        return _ratio_neg(_dft_parameterize(z_c), beta=50.0)
    z_flat = np.concatenate([z0.real, z0.imag])
    result = minimize(objective, z_flat, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    z_opt = result.x[:N] + 1j * result.x[N:]
    h_opt = _dft_parameterize(z_opt)
    return h_opt, _ratio(h_opt)

def _mask_heights(mask):
    """h = |IDFT(mask)|^2 for a binary frequency mask (DC always on)."""
    H = np.zeros(N, dtype=complex)
    H[np.asarray(mask, dtype=int)] = 1.0
    h = np.abs(np.fft.ifft(H)) ** 2
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h

def _mask_score(mask):
    return _ratio(_mask_heights(mask))

def _greedy_mask(K, budget_s):
    """Greedily pick K-1 non-DC bins minimizing the autoconv peak (maximizing ratio)."""
    t0 = time.time()
    chosen = [0]  # DC always active
    candidates = list(range(1, N))
    while len(chosen) < K and (time.time() - t0) < budget_s:
        best_bin = None
        best_s = -1.0
        for b in candidates:
            m = list(chosen) + [b]
            s = _mask_score(m)
            if s > best_s:
                best_s = s
                best_bin = b
        if best_bin is None:
            break
        chosen.append(best_bin)
        candidates.remove(best_bin)
    return sorted(chosen)

def _two_disjoint_gaussians(start_time, max_iter=200):
    def objective(params):
        c1, w1, c2, w2, a1, a2 = params
        w1 = max(abs(w1), 0.01); w2 = max(abs(w2), 0.01)
        a1 = max(a1, 0.0); a2 = max(a2, 0.0)
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        h = (a1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
             + a2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2)))
        return _ratio_neg(h, beta=50.0)
    inits = [
        (-0.15, 0.05, 0.15, 0.05, 1.0, 1.0),
        (-0.1, 0.04, 0.1, 0.04, 1.0, 1.0),
        (-0.2, 0.06, 0.2, 0.06, 1.0, 1.0),
        (-0.15, 0.03, 0.15, 0.08, 1.0, 1.0),
        (-0.15, 0.08, 0.15, 0.03, 1.0, 1.0),
        (-0.1, 0.05, 0.2, 0.04, 1.5, 0.8),
        (-0.05, 0.04, 0.15, 0.04, 1.0, 1.0),
    ]
    best_score = 0.0
    best_h = np.zeros(N)
    for init in inits:
        if _time_left(start_time) < 5.0:
            break
        try:
            result = minimize(objective, init, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'gtol': 1e-10, 'maxls': 30})
            c1, w1, c2, w2, a1, a2 = result.x
            w1 = max(abs(w1), 0.01); w2 = max(abs(w2), 0.01)
            a1 = max(a1, 0.0); a2 = max(a2, 0.0)
            x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
            h = np.maximum(a1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
                           + a2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2)), 0.0)
            s = _ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            continue
    return best_h, best_score

def construct():
    start_time = time.time()

    # Baseline
    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # ---- New structural family: sparse frequency masks ----
    # DC-only mask gives the uniform (baseline) profile.
    for K in [2, 3, 4, 5, 6, 8]:
        if _time_left(start_time) < 12.0:
            break
        budget = min(6.0, max(1.0, _time_left(start_time) * 0.25))
        try:
            mask = _greedy_mask(K, budget)
        except Exception:
            continue
        h_mask = _mask_heights(mask)
        s_mask = _ratio(h_mask)
        if s_mask > best_score:
            best_score = s_mask
            best_h = h_mask.copy()
        # Refine the mask-derived heights directly (removes the sparsity constraint)
        if _time_left(start_time) < 8.0:
            continue
        try:
            h_ref, s_ref = _local_optimize(h_mask, max_iter=150, tol=1e-10)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref.copy()
        except Exception:
            pass

    # ---- Diverse profile initializations (kept from current design) ----
    def _gaussian(center, width, height=1.0):
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    inits = [np.ones(N)]
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
            inits.append(_gaussian(center, width, 1.0))
    rng = np.random.RandomState(42)
    for _ in range(8):
        inits.append(rng.random(N))

    scored = sorted([(_ratio(np.maximum(h, 0.0)), h) for h in inits],
                    key=lambda t: -t[0])
    for s_i, h_i in scored[:6]:
        if _time_left(start_time) < 8.0:
            break
        try:
            h_opt, s_opt = _local_optimize(h_i, max_iter=150, tol=1e-10)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # ---- Two disjoint Gaussians (kept) ----
    if _time_left(start_time) > 12.0:
        try:
            h_tg, s_tg = _two_disjoint_gaussians(start_time, max_iter=150)
            if s_tg > best_score:
                best_score = s_tg
                best_h = h_tg.copy()
        except Exception:
            pass

    # ---- Final high-precision polish (kept) ----
    if _time_left(start_time) > 10.0:
        try:
            h_pol, s_pol = _local_optimize(best_h, max_iter=400, tol=1e-12)
            if s_pol > best_score:
                best_score = s_pol
                best_h = h_pol.copy()
        except Exception:
            pass
        if _time_left(start_time) > 5.0:
            try:
                z_pol = _dft_inverse(best_h)
                h_d, s_d = _optimize_from_z(z_pol, max_iter=150, tol=1e-10)
                if s_d > best_score:
                    best_score = s_d
                    best_h = h_d.copy()
            except Exception:
                pass

    return np.maximum(best_h, 0.0).tolist()
