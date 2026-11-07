import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0
m = 2 * N - 1          # 99
m_plus_1 = m + 1       # 100

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0

    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    g = np.maximum(g, 0.0)   # clip tiny negatives from numerical noise

    sum_g_sq = np.sum(g * g)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    if max_g == 0 or sum_g == 0:
        return 0.0

    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * max_g
    return numerator / denominator


def compute_value_smooth(h, smoothing=1e-4):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0

    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    g = np.maximum(g, 0.0)

    sum_g_sq = np.sum(g * g)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)

    max_g = np.max(g)
    if max_g > 0:
        # log-sum-exp soft max
        shifted = (g - max_g) / smoothing
        # clip for numerical stability
        shifted = np.clip(shifted, -50.0, 50.0)
        smooth_max = max_g + smoothing * np.log(np.sum(np.exp(shifted)))
    else:
        smooth_max = 0.0

    if smooth_max == 0 or sum_g == 0:
        return 0.0

    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    return numerator / denominator


def neg_value(log_h):
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)


def neg_value_smooth(log_h, smoothing=1e-4):
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_smooth(h, smoothing)


# ---------------- Parametric families ----------------
def _x():
    return np.arange(N, dtype=float)

def _c():
    return (N - 1) / 2.0

def parametric_h_gaussian(params):
    sigma = np.exp(params[0])
    x = _x(); c = _c()
    return np.exp(-((x - c)**2) / (2.0 * sigma**2))

def parametric_h_super_gaussian(params):
    sigma = np.exp(params[0])
    beta = params[1]
    x = _x(); c = _c()
    dist = np.abs(x - c)
    return np.exp(-(dist**beta) / (2.0 * (sigma**beta)))

def parametric_h_polynomial(params):
    R = np.exp(params[0])
    p = params[1]
    x = _x(); c = _c()
    t = (x - c) / R
    h = np.maximum(1.0 - t**2, 0.0)
    return h**p

def parametric_h_flat_top(params):
    w_width = np.exp(params[0])
    x = _x(); c = _c()
    d = np.abs(x - c)
    half_w = w_width / 2.0
    h = np.zeros(N)
    mask_inner = d < half_w
    mask_edge = (d >= half_w) & (d <= w_width)
    h[mask_inner] = 1.0
    d_edge = d[mask_edge]
    t = (d_edge - half_w) / (w_width - half_w)
    h[mask_edge] = 0.5 * (1.0 + np.cos(np.pi * t))
    return h

def parametric_h_hamming(params):
    # h_i = 0.54 + 0.46*cos(2*pi*(i-c)/(2*L)) clipped to support
    L = np.exp(params[0])
    x = _x(); c = _c()
    t = (x - c) / L
    h = 0.54 + 0.46 * np.cos(np.pi * t)
    h = np.where(np.abs(t) <= 1.0, h, 0.0)
    return np.maximum(h, 0.0)

def parametric_h_blackman(params):
    L = np.exp(params[0])
    x = _x(); c = _c()
    t = (x - c) / L
    h = 0.42 - 0.5*np.cos(np.pi*t) + 0.08*np.cos(2*np.pi*t)
    h = np.where(np.abs(t) <= 1.0, h, 0.0)
    return np.maximum(h, 0.0)

def parametric_h_cosine(params):
    # h_i = cos(pi*(i-c)/(2*L))_+
    L = np.exp(params[0])
    x = _x(); c = _c()
    t = (x - c) / (2.0 * L)
    h = np.cos(np.pi * t)
    h = np.where(np.abs(t) <= 1.0, h, 0.0)
    return np.maximum(h, 0.0)

def parametric_h_two_bumps(params):
    sigma = np.exp(params[0])
    sep = params[1]
    x = _x(); c = _c()
    h = np.exp(-((x - (c - sep))**2) / (2.0*sigma**2)) + \
        np.exp(-((x - (c + sep))**2) / (2.0*sigma**2))
    return h


def optimize_parametric(func, bounds, deadline):
    def obj(params):
        try:
            h = func(params)
            if np.any(h < 0) or np.any(~np.isfinite(h)) or np.max(h) == 0:
                return 1e6
            return -compute_value(h)
        except Exception:
            return 1e6

    try:
        res = differential_evolution(
            obj, bounds, seed=42, maxiter=150, tol=1e-8, polish=True,
            init='sobol'
        )
        best_params = res.x
        best_h = func(best_params)
        best_val = compute_value(best_h)
        return best_val, best_h, best_params
    except Exception:
        return 0.0, None, None


def local_refine(h_start, deadline):
    """Refine h_start in log-space using smooth then exact objective."""
    h_start = np.maximum(h_start, 1e-8)
    log_h_start = np.log(h_start)

    # Phase 1: smooth objective, moderate tolerance
    try:
        res = minimize(
            neg_value_smooth, log_h_start, args=(1e-3,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        h1 = np.exp(res.x - np.mean(res.x))
        h1 = np.maximum(h1, 0.0)
        v1 = compute_value(h1)
    except Exception:
        h1 = h_start.copy()
        v1 = compute_value(h1)

    # Phase 2: exact objective, tighter tolerance
    log_h1 = np.log(np.maximum(h1, 1e-8))
    try:
        res2 = minimize(
            neg_value, log_h1,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        h2 = np.exp(res2.x - np.mean(res2.x))
        h2 = np.maximum(h2, 0.0)
        v2 = compute_value(h2)
        if v2 >= v1:
            return h2, v2
        return h1, v1
    except Exception:
        return h1, v1


def _coordinate_descent_polish(h, max_rounds=20, tol=1e-12):
    """Coordinate descent: optimize each coordinate while holding others fixed."""
    h = np.maximum(h, 1e-8).copy()
    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            # Try small perturbations around current value
            best_local = h[i]
            best_local_val = compute_value(h)
            for scale in [0.5, 0.7, 1.3, 2.0, 3.0]:
                for sign in [-1, 1]:
                    h_trial = h.copy()
                    h_trial[i] = max(h[i] * scale if sign == 1 else h[i] / scale, 1e-8)
                    v = compute_value(h_trial)
                    if v > best_local_val:
                        best_local_val = v
                        best_local = h_trial[i]
            h[i] = best_local
            if abs(h[i] - best_local) > tol:
                improved = True
        if not improved:
            break
    return h


def _multi_start_lbfgsb(h_start, n_starts=5, rng_seed=42):
    """Run multiple L-BFGS-B starts from perturbations of h_start."""
    rng = np.random.default_rng(rng_seed)
    best_h = h_start.copy()
    best_val = compute_value(h_start)
    
    for i in range(n_starts):
        if time.time() > deadline_global - 10:
            break
        # Different perturbation scales
        sigma = 0.1 * (i + 1)
        log_h = np.log(np.maximum(h_start, 1e-8))
        log_h_pert = log_h + rng.normal(0, sigma, size=N)
        h_pert = np.exp(log_h_pert - np.mean(log_h_pert))
        h_pert = np.maximum(h_pert, 1e-8)
        
        try:
            res = minimize(
                neg_value, np.log(np.maximum(h_pert, 1e-8)),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = np.exp(res.x - np.mean(res.x))
            h_opt = np.maximum(h_opt, 1e-8)
            v = compute_value(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    return best_h, best_val


deadline_global = 0.0  # Set in construct()

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 135.0  # 15s safety margin

    best_val = 0.0
    best_h = np.ones(N)
    seed_val = compute_value(best_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = best_h.copy()

    # --- Generate diverse initial candidates ---
    x = _x(); c = _c()
    candidates = [np.ones(N)]
    
    # Parametric families (fast evaluation, no DE)
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        candidates.append(parametric_h_gaussian([np.log(sigma)]))
    for sigma in [2.0, 3.0, 5.0]:
        for beta in [1.5, 2.0, 3.0, 4.0]:
            candidates.append(parametric_h_super_gaussian([np.log(sigma), beta]))
    for R in [10.0, 15.0, 20.0, 25.0]:
        for p in [1.0, 2.0, 3.0]:
            candidates.append(parametric_h_polynomial([np.log(R), p]))
    for L in [10.0, 15.0, 20.0]:
        candidates.append(parametric_h_hamming([np.log(L)]))
        candidates.append(parametric_h_blackman([np.log(L)]))
        candidates.append(parametric_h_cosine([np.log(L)]))
    for sigma in [2.0, 4.0]:
        for sep in [5.0, 10.0, 15.0]:
            candidates.append(parametric_h_two_bumps([np.log(sigma), sep]))
    
    # Heuristic shapes
    h_tri = np.maximum(1.0 - np.abs(x - c) / c, 0.0)
    h_rc = 0.5 * (1.0 + np.cos(np.pi * (x - c) / c))
    h_sine = np.sin(np.pi * (x / (N - 1)))
    h_2b = np.exp(-((x - (c - 5))**2) / 5) + np.exp(-((x - (c + 5))**2) / 5)
    h_plateau = np.where(np.abs(x - c) <= 10.0, 1.0, 0.0)
    h_plateau_smooth = np.exp(-((np.abs(x - c) - 10.0).clip(0, None)**2) / 4.0)
    for hc in [h_tri, h_rc, h_sine, h_2b, h_plateau, h_plateau_smooth]:
        candidates.append(hc)

    # Random initializations
    rng = np.random.default_rng(12345)
    for _ in range(10):
        h_rand = rng.random(N) * 0.5 + 0.5
        candidates.append(h_rand)

    # --- Refine each candidate with local_refine ---
    for h_cand in candidates:
        if time.time() > deadline_global - 20.0:
            break
        h_ref, v_ref = local_refine(h_cand, deadline_global)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref.copy()

    # --- Coordinate descent polish on best ---
    if time.time() < deadline_global - 10.0:
        h_cd = _coordinate_descent_polish(best_h, max_rounds=30)
        v_cd = compute_value(h_cd)
        if v_cd > best_val:
            best_val = v_cd
            best_h = h_cd.copy()

    # --- Multi-start L-BFGS-B around best ---
    if time.time() < deadline_global - 10.0:
        h_ms, v_ms = _multi_start_lbfgsb(best_h, n_starts=8, rng_seed=999)
        if v_ms > best_val:
            best_val = v_ms
            best_h = h_ms.copy()

    # --- Final high-precision polish ---
    if time.time() < deadline_global - 5.0:
        log_h_best = np.log(np.maximum(best_h, 1e-8))
        try:
            res = minimize(
                neg_value, log_h_best,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'maxfun': 30000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            h_final = np.exp(res.x - np.mean(res.x))
            h_final = np.maximum(h_final, 0.0)
            val_final = compute_value(h_final)
            if val_final > best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Ensure output validity
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)

    return list(best_h)


if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Value: {val}")
    print(f"Height sample: {h[:5]}")
