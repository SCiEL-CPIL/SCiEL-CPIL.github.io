# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of support points.
    weights: 1D array of weights (will be normalized).
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Precompute pairwise sums s_i + s_j
    sums = support[:, None] + support[None, :]
    # Precompute outer product of weights
    w_outer = w[:, None] * w[None, :]
    
    P = 0.0
    # Loop over X4 (index l) and X1 (index k)
    for l in range(n):
        wl = w[l]
        if wl < 1e-12:
            continue
        for k in range(n):
            wk = w[k]
            if wk < 1e-12:
                continue
            target = 2 * support[l] - support[k]
            # Sum of w_i * w_j for pairs (i, j) such that s_i + s_j < target
            mask = sums < target
            prob_inner = np.sum(w_outer * mask)
            P += wl * wk * prob_inner
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # 1. Analytic 2-point solution
    # Maximize (1-p) * p^2 * (3 - 2p)
    # p* = (15 - sqrt(33)) / 16
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    support_2pt = np.array([0.0, 1.0], dtype=np.float64)
    weights_2pt = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
    
    P_2pt = calc_P(support_2pt, weights_2pt)
    best_P = P_2pt
    best_support = support_2pt
    best_weights = weights_2pt
    
    # 2. Optimization for n > 2
    rng = np.random.default_rng(42)
    
    # Try different numbers of points
    n_values = [3, 4, 5, 6, 8, 10, 12, 15, 20]
    
    for n in n_values:
        if time.time() > deadline:
            break
            
        # Number of restarts: more for smaller n
        num_restarts = 10 if n <= 5 else (5 if n <= 10 else 3)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Parameterization: log_weights, log_support
            # x = [log_w_0, ..., log_w_{n-1}, log_s_0, ..., log_s_{n-1}]
            
            if r == 0:
                # Start near the 2-point solution extended with small weights
                # Weights: p_opt, 1-p_opt, and small epsilon for others
                # Support: 0, 1, and distinct values
                log_w_init = np.zeros(n)
                log_s_init = np.zeros(n)
                
                # Set first two to match 2pt solution
                log_w_init[0] = np.log(p_opt)
                log_w_init[1] = np.log(1.0 - p_opt)
                for i in range(2, n):
                    log_w_init[i] = np.log(1e-3)
                
                log_s_init[0] = np.log(1e-6) # approx 0
                log_s_init[1] = np.log(1.0)
                for i in range(2, n):
                    log_s_init[i] = np.log(1.0 + i * 0.5)
            else:
                # Random initialization
                raw_w = rng.exponential(size=n)
                log_w_init = np.log(raw_w)
                # Support in [0.1, 10]
                log_s_init = np.log(rng.uniform(0.1, 10.0, size=n))
            
            x0 = np.concatenate([log_w_init, log_s_init])
            
            def objective(x):
                log_w, log_s = np.split(x, [n])
                # Softmax for weights to ensure sum=1 and positive
                # Subtract max for numerical stability
                lw_max = np.max(log_w)
                w_unnorm = np.exp(log_w - lw_max)
                w = w_unnorm / np.sum(w_unnorm)
                
                # Exp for support to ensure positive
                s = np.exp(log_s)
                
                return -calc_P(s, w)
            
            # Bounds for log space
            # log_w in [-20, 5] (weights from 2e-9 to 148)
            # log_s in [-10, 5] (support from 4.5e-5 to 148)
            bounds = [(-20, 5)] * n + [(-10, 5)] * n
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               bounds=bounds,
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                
                log_w, log_s = np.split(res.x, [n])
                lw_max = np.max(log_w)
                w_unnorm = np.exp(log_w - lw_max)
                w = w_unnorm / np.sum(w_unnorm)
                s = np.exp(log_s)
                
                # Ensure validity
                s = np.clip(s, 1e-9, None)
                w = np.clip(w, 0, None)
                w_sum = np.sum(w)
                if w_sum > 1e-9:
                    w = w / w_sum
                else:
                    continue
                    
                P_val = calc_P(s, w)
                
                if P_val > best_P:
                    best_P = P_val
                    best_support = s
                    best_weights = w
                    
            except Exception:
                pass
                
    return best_support, best_weights

# EVOLVE-BLOCK-END
