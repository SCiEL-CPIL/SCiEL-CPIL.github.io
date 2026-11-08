import time
import numpy as np
from scipy.optimize import minimize

N = 50
_start_time = time.time()
_TIMEOUT = 110  # safety margin before 120s limit

def _ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    linf = np.max(conv)
    if linf < 1e-15:
        return 0.0
    return l2_sq / (l1 * linf)

def _ratio_smooth(h, alpha=1.0):
    """Smooth surrogate for the ratio using log-sum-exp for soft-max."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    # Soft-max via log-sum-exp
    c = np.log(np.sum(np.exp(alpha * conv))) / alpha
    return l2_sq / (l1 * np.exp(c))

def _neg_ratio_smooth(params, alpha=1.0):
    h = np.abs(params)
    h_max = np.max(h)
    if h_max < 1e-15:
        return 0.0
    h = h / h_max
    return -_ratio_smooth(h, alpha)

def construct():
    t0 = time.time()
    deadline = t0 + _TIMEOUT
    
    best_h = [0.0] * N
    best_h[0] = 1e-12
    best_score = _ratio(best_h)
    
    # Try 2-point cluster with grid search over ratios
    for r in np.linspace(0.2, 5.0, 30):
        h = [0.0] * N
        h[0] = r
        h[1] = 1.0
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h
    
    # Try 3-point clusters with grid search
    for r1 in np.linspace(0.5, 3.0, 10):
        for r2 in np.linspace(0.5, 3.0, 10):
            if time.time() > deadline - 30:
                break
            h = [0.0] * N
            h[0] = r1
            h[1] = 1.0
            h[2] = r2
            s = _ratio(h)
            if s > best_score:
                best_score = s
                best_h = h
    
    # Try 4-point clusters with grid search
    for r1 in [1.0, 1.5, 2.0]:
        for r2 in [0.5, 1.0, 1.5]:
            for r3 in [0.5, 1.0, 1.5]:
                if time.time() > deadline - 20:
                    break
                h = [0.0] * N
                h[0] = r1
                h[1] = 1.0
                h[2] = r2
                h[3] = r3
                s = _ratio(h)
                if s > best_score:
                    best_score = s
                    best_h = h
    
    # Use scipy to optimize with smooth surrogate
    try:
        # Start from best found so far
        h_start = np.array(best_h, dtype=float)
        h_max = np.max(h_start)
        if h_max > 1e-15:
            h_start = h_start / h_max
        else:
            h_start = np.zeros(N)
            h_start[0] = 1.0
        
        # Multiple trials with different initializations and alpha values
        for trial in range(15):
            if time.time() > deadline - 15:
                break
            
            if trial < 5:
                # Start from best found
                x0 = h_start.copy()
            elif trial < 10:
                # Random sparse initialization
                x0 = np.zeros(N)
                n_points = np.random.randint(2, 6)
                indices = np.random.choice(N, n_points, replace=False)
                x0[indices] = np.random.uniform(0.1, 1.0, n_points)
            else:
                # Random dense initialization
                x0 = np.random.random(N) * 0.5
                x0[0] = 1.0
            
            # Use smooth surrogate for optimization
            alpha = 1.0 if trial < 8 else 2.0
            res = minimize(_neg_ratio_smooth, x0, args=(alpha,), 
                          method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            h_opt = np.abs(res.x)
            h_max = np.max(h_opt)
            if h_max > 1e-15:
                h_opt = h_opt / h_max
            s = _ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = list(h_opt)
                h_start = h_opt.copy()
        
    except Exception:
        pass
    
    # Final polish with tighter tolerance on the exact ratio
    try:
        if time.time() < deadline - 5:
            h_start = np.array(best_h, dtype=float)
            h_max = np.max(h_start)
            if h_max > 1e-15:
                h_start = h_start / h_max
            
            def neg_ratio_exact(params):
                h = np.abs(params)
                h_max = np.max(h)
                if h_max < 1e-15:
                    return 0.0
                h = h / h_max
                return -_ratio(h)
            
            res = minimize(neg_ratio_exact, h_start, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            h_opt = np.abs(res.x)
            h_max = np.max(h_opt)
            if h_max > 1e-15:
                h_opt = h_opt / h_max
            s = _ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = list(h_opt)
    except Exception:
        pass
    
    return best_h
