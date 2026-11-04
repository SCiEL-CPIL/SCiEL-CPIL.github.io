import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 140.0  # Safety margin before 150s kill
_start_time = None

def _time_left():
    global _start_time
    if _start_time is None:
        return _TIME_LIMIT
    return _TIME_LIMIT - (time.time() - _start_time)

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        # Check if i can be added
        # New sums will be i+s for s in S, and 2*i
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _optimize_weights(sidon_indices, h_init, max_time=30.0):
    """Optimize weights on Sidon set positions."""
    K = len(sidon_indices)
    if K == 0:
        return h_init
    
    # Work in log-space to ensure positivity
    h0 = np.zeros(N)
    for i, idx in enumerate(sidon_indices):
        h0[idx] = h_init[idx] if h_init[idx] > 1e-15 else 1.0/K
    
    log_w0 = np.log(h0[sidon_indices])
    
    def objective(log_w):
        if _time_left() < 5.0:
            return 0.0
        w = np.exp(log_w)
        h = np.zeros(N)
        for i, idx in enumerate(sidon_indices):
            h[idx] = w[i]
        return -_compute_ratio(h)
    
    try:
        res = minimize(objective, log_w0, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < objective(log_w0):
            w = np.exp(res.x)
            h = np.zeros(N)
            for i, idx in enumerate(sidon_indices):
                h[idx] = w[i]
            # Normalize to L1=1
            l1 = np.sum(h)
            if l1 > 1e-15:
                h = h / l1
            return h
    except Exception:
        pass
    return h_init

def _local_optimize(h, max_time=20.0):
    """Local optimization polish on the full vector."""
    # Work in log-space for unconstrained optimization
    h_safe = np.maximum(h, 1e-15)
    log_h0 = np.log(h_safe)
    
    def objective(log_h):
        if _time_left() < 5.0:
            return 0.0
        h = np.exp(log_h)
        return -_compute_ratio(h)
    
    try:
        res = minimize(objective, log_h0, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < objective(log_h0):
            h_opt = np.exp(res.x)
            # Enforce non-negativity (should already be positive from exp)
            h_opt = np.maximum(h_opt, 0.0)
            l1 = np.sum(h_opt)
            if l1 > 1e-15:
                h_opt = h_opt / l1
            return h_opt
    except Exception:
        pass
    return h

def construct():
    global _start_time
    _start_time = time.time()
    
    # Try to find a larger Sidon set
    best_h = None
    best_ratio = 0.0
    
    # Try different Sidon set sizes
    for K in [8, 9, 10, 11, 12]:
        if _time_left() < 30.0:
            break
        
        sidon_indices = _get_sidon_set(K, N - 1)
        if len(sidon_indices) < K:
            continue
        
        # Initialize with equal weights
        h = np.zeros(N)
        for idx in sidon_indices:
            h[idx] = 1.0 / K
        
        # Optimize weights
        h_opt = _optimize_weights(sidon_indices, h, max_time=15.0)
        
        ratio = _compute_ratio(h_opt)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h_opt.copy()
    
    # Fallback if no Sidon set worked
    if best_h is None:
        sidon_indices = _get_sidon_set(8, N - 1)
        best_h = np.zeros(N)
        for idx in sidon_indices:
            best_h[idx] = 1.0 / len(sidon_indices)
    
    # Final local polish
    if _time_left() > 15.0:
        best_h = _local_optimize(best_h, max_time=10.0)
    
    # Ensure non-negative and normalize
    best_h = np.maximum(best_h, 0.0)
    l1 = np.sum(best_h)
    if l1 > 1e-15:
        best_h = best_h / l1
    
    return [float(x) for x in best_h]
