import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv(h):
    """Full linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=100.0):
    """Compute the ratio using log-sum-exp for smooth max."""
    ac = _autoconv(h)
    # log-sum-exp for max
    max_ac = np.max(ac)
    if max_ac <= 0:
        return 0.0
    # Smooth max
    log_sum_exp = np.log(np.sum(np.exp(beta * ac - beta * max_ac))) + beta * max_ac
    smooth_max = log_sum_exp / beta
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 <= 0 or smooth_max <= 0:
        return 0.0
    return l2_sq / (l1 * smooth_max)

def _neg_ratio_smooth(x, beta=100.0):
    """Negative ratio for minimization. x are log-softmax parameters."""
    # Softmax
    x_shifted = x - np.max(x)
    h = np.exp(x_shifted)
    h /= np.sum(h)
    return -_ratio(h, beta)

def construct():
    deadline = time.time() + 140.0  # 10s margin
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    # Try different initializations and optimizations
    rng = np.random.RandomState(42)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        x0 = rng.randn(N) * 2.0
        res = minimize(_neg_ratio_smooth, x0, args=(100.0,), method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.success or res.fun < 0:
            x_shifted = res.x - np.max(res.x)
            h = np.exp(x_shifted)
            h /= np.sum(h)
            score = _ratio(h)
            if score > best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline - 15.0:
            break
    
    # Final polish: use the best h as initialization with higher beta
    if time.time() < deadline - 5.0:
        x_best = np.log(best_h + 1e-30)
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            res = minimize(_neg_ratio_smooth, x_best, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < 0:
                x_shifted = res.x - np.max(res.x)
                h = np.exp(x_shifted)
                h /= np.sum(h)
                score = _ratio(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Return as list of floats
    return list(best_h)
