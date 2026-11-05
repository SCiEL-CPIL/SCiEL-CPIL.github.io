import numpy as np
import scipy.optimize as opt
import time

N = 400

def _compute_value(h):
    S = np.sum(h)
    if S == 0 or not np.all(np.isfinite(h)):
        return np.inf
    R = np.convolve(h, h)
    return abs(2.0 * N * np.max(R) / (S * S))

def _chirp_h(params):
    t = np.arange(N, dtype=np.float64)
    freq, phase, amp = params[0], params[1], params[2]
    h1f, h1a = params[3], params[4]
    h2f, h2a = params[5], params[6]
    offset, scale = params[7], params[8]
    h = amp * np.cos(2 * np.pi * freq * t / N + phase)
    h += h1a * np.sin(2 * np.pi * h1f * t / N)
    h += h2a * np.sin(2 * np.pi * h2f * t / N)
    h += offset
    h *= scale
    return h

def _obj_params(params):
    h = _chirp_h(params)
    return _compute_value(h)

def _random_search_best(n_trials=300, deadline=None):
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_params = None
    for i in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        params = np.array([
            rng.uniform(0.01, 5.0),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 2.0),
            rng.uniform(0.01, 5.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.01, 5.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.5, 2.0),
        ])
        val = _obj_params(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
    return best_params, best_val

def _local_refine(h, deadline=None):
    def _smooth_obj(x):
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        R = np.convolve(x, x)
        temp = 0.1
        shifted = R / temp
        max_shifted = np.max(shifted)
        logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_max = temp * logsumexp
        return abs(2.0 * N * smooth_max / (S * S))
    try:
        result = opt.minimize(_smooth_obj, h, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12})
        if result.success and _compute_value(result.x) < _compute_value(h):
            return result.x
    except Exception:
        pass
    def _direct_obj(x):
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        R = np.convolve(x, x)
        return abs(2.0 * N * np.max(R) / (S * S))
    try:
        result = opt.minimize(_direct_obj, h, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10})
        if result.success and _compute_value(result.x) < _compute_value(h):
            return result.x
    except Exception:
        pass
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    best_params, best_val = _random_search_best(n_trials=300, deadline=deadline - 60)
    if best_params is None:
        return np.ones(N)
    best_h = _chirp_h(best_params)
    best_value = _compute_value(best_h)
    remaining = deadline - time.time()
    if remaining > 10:
        refined_h = _local_refine(best_h, deadline=deadline - 5)
        refined_val = _compute_value(refined_h)
        if refined_val < best_value:
            best_h = refined_h
            best_value = refined_val
    remaining = deadline - time.time()
    if remaining > 20:
        rng = np.random.default_rng(123)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = best_h + rng.normal(0, 0.01, N)
            refined = _local_refine(perturbed, deadline=deadline - 5)
            val = _compute_value(refined)
            if val < best_value:
                best_h = refined
                best_value = val
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    return best_h
