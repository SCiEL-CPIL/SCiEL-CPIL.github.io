"""Combined design: Gaussian-mixture parametric search (smooth surrogate) + full-dim local polish
for the second autocorrelation inequality value. Entry point: construct() -> 50 heights."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0
DEADLINE_OFFSET = TIME_LIMIT - SAFETY  # 140s

# ---- fast evaluator (mirrors the contract exactly) ----
def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

# ---- Gaussian mixture model: 3K params -> 50 nonnegative heights ----
def _gmm_from_params(params, K=4):
    h = np.zeros(N)
    x = np.linspace(0.0, 1.0, N)
    for k in range(K):
        A = params[3 * k]
        c = params[3 * k + 1]
        s = params[3 * k + 2]
        if s < 1e-6:
            s = 1e-6
        h += A * np.exp(-((x - c) ** 2) / (2.0 * s ** 2))
    return np.maximum(h, 0.0)

def _soft_norm_inf(g, alpha=50.0):
    abs_g = np.abs(g)
    if len(abs_g) == 0:
        return 0.0
    max_val = np.max(abs_g)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_g - max_val
    log_sum = max_val + np.log(np.sum(np.exp(alpha * shifted)) / alpha)
    return log_sum

def _neg_value_gmm_smooth(params, K=4, alpha=50.0):
    try:
        h = _gmm_from_params(params, K)
        if not np.all(np.isfinite(h)):
            return 1e10
        g = np.convolve(h, h)
        m = len(g)
        y = np.concatenate(([0.0], g, [0.0]))
        w = 1.0 / (m + 1)
        a = y[:-1]
        b = y[1:]
        l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        if norm_1 < 1e-12:
            return 1e10
        smooth_inf = _soft_norm_inf(g, alpha)
        if smooth_inf < 1e-12:
            return 1e10
        return -(l2 / (norm_1 * smooth_inf))
    except Exception:
        return 1e10

def _neg_value_gmm_exact(params, K=4):
    try:
        h = _gmm_from_params(params, K)
        if not np.all(np.isfinite(h)):
            return 1e10
        return -_value_from_h(h)
    except Exception:
        return 1e10

def _neg_value_full(h):
    h = np.clip(np.asarray(h, dtype=float), 0.0, None)
    try:
        if not np.all(np.isfinite(h)):
            return 1e10
        return -_value_from_h(h)
    except Exception:
        return 1e10

def _make_initial_guesses(rng, K, n_guesses):
    guesses = []
    for i in range(n_guesses):
        params = np.zeros(3 * K)
        for k in range(K):
            if i < n_guesses // 3:
                # standard Gaussian
                params[3 * k] = rng.uniform(0.1, 3.0)
                params[3 * k + 1] = rng.uniform(0.0, 1.0)
                params[3 * k + 2] = rng.uniform(0.02, 0.4)
            elif i < 2 * n_guesses // 3:
                # spike-like
                params[3 * k] = rng.uniform(1.0, 5.0)
                params[3 * k + 1] = rng.uniform(0.0, 1.0)
                params[3 * k + 2] = rng.uniform(0.01, 0.08)
            else:
                # flatter
                params[3 * k] = rng.uniform(0.5, 1.5)
                params[3 * k + 1] = rng.uniform(0.0, 1.0)
                params[3 * k + 2] = rng.uniform(0.15, 0.45)
        guesses.append(params)
    return guesses

def _global_search(start_time, best_h, best_val, K=4):
    rng = np.random.default_rng(12345)
    dim = 3 * K
    bounds = [(0.01, 5.0), (0.0, 1.0), (0.005, 0.5)] * K
    n_restarts = 80
    guesses = _make_initial_guesses(rng, K, n_restarts)
    
    for i, params0 in enumerate(guesses):
        if time.time() - start_time > DEADLINE_OFFSET - 30:
            break
        try:
            res = minimize(_neg_value_gmm_smooth, params0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 80})
            if res.x is not None:
                h_opt = _gmm_from_params(res.x, K)
                val_opt = _value_from_h(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: polish parametric with exact objective
    if time.time() - start_time < DEADLINE_OFFSET - 15:
        best_params = None
        for trial in range(5):
            if time.time() - start_time > DEADLINE_OFFSET - 15:
                break
            if best_params is None:
                p_init = np.zeros(dim)
                for k in range(K):
                    p_init[3 * k] = 1.0
                    p_init[3 * k + 1] = 0.5
                    p_init[3 * k + 2] = 0.2
            else:
                p_init = best_params + rng.normal(0, 0.05, size=dim)
                p_init = np.clip(p_init, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                res2 = minimize(_neg_value_gmm_exact, p_init, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
                if res2.x is not None:
                    h_opt = _gmm_from_params(res2.x, K)
                    val_opt = _value_from_h(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        best_params = res2.x.copy()
            except Exception:
                pass
    
    return best_h, best_val

def _local_polish(start_time, h0, max_val):
    rng = np.random.default_rng(999)
    best_h = h0.copy()
    best_val = max_val
    polish_starts = 8
    for k in range(polish_starts):
        if time.time() - start_time > DEADLINE_OFFSET - 5:
            break
        if k == 0:
            h = h0.copy()
        else:
            h = h0 + rng.normal(0.0, 0.05, N)
            h = np.clip(h, 0.0, None)
        try:
            res = minimize(_neg_value_full, h, method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7})
            h2 = np.clip(res.x, 0.0, None)
            v = _value_from_h(h2)
            if v > best_val:
                best_val = v
                best_h = h2.copy()
        except Exception:
            pass
    return best_h, best_val

def construct():
    start_time = time.time()
    seed_h = np.ones(N)
    best_h = seed_h.copy()
    best_val = _value_from_h(seed_h)
    
    # Global parametric search with GMM
    best_h, best_val = _global_search(start_time, best_h, best_val, K=4)
    
    # Local polish on full dimension
    best_h, best_val = _local_polish(start_time, best_h, best_val)
    
    # Final safety
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    return list(best_h)
