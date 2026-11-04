import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def _elapsed(start):
    return time.time() - start

def _time_left(start, deadline=140):
    return deadline - _elapsed(start)

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _ratio(h):
    """
    Compute ||h||_2^2 / (||h||_1 * max(conv(h)))
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return 0.0
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    conv = _autoconv(h)
    c_max = np.max(conv)
    if c_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * c_max)

def _ratio_neg(h):
    return -_ratio(h)

def _smooth_ratio(h, beta):
    """
    Smooth approximation of the ratio using log-sum-exp for the max.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return -1e10
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1e10
    l2_sq = np.sum(h * h)
    
    conv = _autoconv(h)
    c_max = np.max(conv)
    if c_max < 1e-15:
        return -1e10
    
    conv_shifted = conv - c_max
    sum_exp = np.sum(np.exp(beta * conv_shifted))
    log_max_approx = c_max + (1.0 / beta) * np.log(sum_exp)
    
    if log_max_approx < 1e-15:
        return -1e10
        
    ratio = l2_sq / (l1 * log_max_approx)
    return ratio

def _gaussian_profile(width, center, amplitude=1.0):
    """
    Create a Gaussian profile on the grid.
    """
    x = np.linspace(-0.25, 0.25, N)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def _lorentzian_profile(width, center, amplitude=1.0):
    """
    Create a Lorentzian profile on the grid.
    """
    x = np.linspace(-0.25, 0.25, N)
    return amplitude * (width ** 2) / ((x - center) ** 2 + width ** 2)

def _power_profile(power, center, amplitude=1.0):
    """
    Create a power-law profile on the grid.
    """
    x = np.linspace(-0.25, 0.25, N)
    base = 1.0 - np.abs(x - center) / 0.25
    base = np.maximum(base, 0)
    return amplitude * (base ** power)

def _optimize_from_z(z, start, deadline=140):
    """
    Optimize from a parameterization z.
    This is a placeholder for more complex parameterizations.
    """
    return z

def _dft_parameterize(h):
    """
    Parameterize h using its DFT coefficients (real and positive).
    """
    H = np.fft.rfft(h)
    return H

def _dft_inverse(H):
    """
    Inverse DFT to get h.
    """
    h = np.fft.irfft(H, n=N)
    return h

def _local_optimize(h, start, deadline=140):
    """
    Perform local optimization on h.
    """
    if _time_left(start, deadline) < 5:
        return h
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    # Try different optimization methods and initializations
    for method in ['SLSQP', 'L-BFGS-B']:
        if _time_left(start, deadline) < 5:
            break
        
        try:
            # Normalize h
            h_norm = h / np.sum(h)
            
            # Define objective
            def objective(x):
                return _ratio_neg(x)
            
            # Constraints: sum(x) = 1, x >= 0
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            bounds = [(0.0, 1.0) for _ in range(N)]
            
            # Optimize
            res = minimize(objective, h_norm, method=method, bounds=bounds,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < -best_score:
                h_new = np.maximum(res.x, 0)
                s = np.sum(h_new)
                if s > 1e-15:
                    h_new /= s
                    score_new = _ratio(h_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_h = h_new
        except Exception:
            pass
    
    return best_h

def _get_sidon_set(n, max_size=10, num_trials=1000):
    """
    Finds a Sidon subset of {0, ..., n-1} of size max_size.
    """
    for size in range(max_size, 2, -1):
        for _ in range(num_trials):
            S = set(random.sample(range(n), size))
            S_list = list(S)
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_list[i] + S_list[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return S_list
    # Fallback to uniform
    return list(range(n))

def _try_larger_sidon(n, current_size, start, deadline=140):
    """
    Try to find a larger Sidon set.
    """
    if _time_left(start, deadline) < 10:
        return None
    
    for size in range(current_size + 1, min(current_size + 5, n)):
        indices = _get_sidon_set(n, max_size=size, num_trials=500)
        if len(indices) == size:
            return indices
    return None

def _optimize_weights(indices, start, deadline=140):
    """
    Optimizes weights on the given indices to maximize the ratio.
    """
    n = N
    M = len(indices)
    if M == 0:
        return np.zeros(n)
    
    # Initial guess: uniform on indices
    h0 = np.zeros(n)
    for idx in indices:
        h0[idx] = 1.0 / M
        
    best_h = h0.copy()
    best_score = _ratio(h0)
    
    # Try different optimization methods
    for method in ['SLSQP', 'L-BFGS-B']:
        if _time_left(start, deadline) < 5:
            break
        
        try:
            # Normalize h
            h_norm = h0 / np.sum(h0)
            
            # Define objective
            def objective(x):
                return _ratio_neg(x)
            
            # Constraints: sum(x) = 1, x >= 0
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            bounds = [(0.0, 1.0) for _ in range(n)]
            
            # Optimize
            res = minimize(objective, h_norm, method=method, bounds=bounds,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < -best_score:
                h_new = np.maximum(res.x, 0)
                s = np.sum(h_new)
                if s > 1e-15:
                    h_new /= s
                    score_new = _ratio(h_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_h = h_new
        except Exception:
            pass
    
    return best_h

def construct():
    start = time.time()
    deadline = 140  # Leave 10s margin for safety (kill is 150s)
    
    # Find a good Sidon set
    indices = None
    for size in range(8, 3, -1):
        if _time_left(start, deadline) < 20:
            break
        indices = _get_sidon_set(N, max_size=size, num_trials=2000)
        if len(indices) == size:
            break
            
    if indices is None or len(indices) < 2:
        # Fallback to uniform
        return [1.0/N] * N
        
    # Optimize weights
    if _time_left(start, deadline) < 5:
        # Not enough time, just return uniform on Sidon set
        h = np.zeros(N)
        for idx in indices:
            h[idx] = 1.0 / len(indices)
        return h.tolist()
        
    h = _optimize_weights(indices, start, deadline)
    
    # Local optimization
    h = _local_optimize(h, start, deadline)
    
    # Ensure non-negative and normalized
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return h.tolist()
