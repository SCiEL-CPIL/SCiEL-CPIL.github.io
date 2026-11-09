import numpy as np
import time
from scipy.optimize import minimize

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,), assumed normalized
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    # Precompute pairwise sums and weights
    sums = s[:, None] + s[None, :]
    w_prod = w[:, None] * w[None, :]
    
    # Sort sums for faster threshold queries
    sums_flat = sums.ravel()
    w_prod_flat = w_prod.ravel()
    sort_idx = np.argsort(sums_flat)
    sums_sorted = sums_flat[sort_idx]
    w_prod_sorted = w_prod_flat[sort_idx]
    cumsum_w = np.cumsum(w_prod_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(sums_sorted, threshold, side='left')
            inner_sum = cumsum_w[idx - 1] if idx > 0 else 0.0
            P += w[k] * w[l] * inner_sum
    return P

def _optimize_weights_for_support(s, n_restarts=5, maxiter=200):
    """
    Optimize weights for a fixed support using multiple Nelder-Mead restarts.
    Returns the best weights found, or None if optimization fails.
    """
    n = len(s)
    best_w = None
    best_P = 0.0
    
    for restart in range(n_restarts):
        if restart == 0:
            # Start from uniform
            x0 = np.ones(n) / n
        else:
            # Random Dirichlet
            x0 = np.random.dirichlet(np.ones(n))
        
        def objective(params):
            w = np.maximum(params, 0.0)
            wsum = w.sum()
            if wsum < 1e-12:
                return 1e10
            w = w / wsum
            return -_calc_P(s, w)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or res.fun < -best_P:
                w_opt = np.maximum(res.x, 0.0)
                wsum = w_opt.sum()
                if wsum > 1e-12:
                    w_opt = w_opt / wsum
                    P_val = _calc_P(s, w_opt)
                    if P_val > best_P:
                        best_P = P_val
                        best_w = w_opt.copy()
        except Exception:
            pass
    
    return best_w

def beat_the_average_measure():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_P = 0.0
    best_support = np.array([0.0])
    best_weights = np.array([1.0])
    
    def try_config(s, w):
        nonlocal best_P, best_support, best_weights
        s = np.asarray(s, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if len(s) != len(w) or len(s) < 1:
            return
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            return
        if np.any(s < -1e-9):
            return
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            return
        if np.any(w < -1e-9):
            return
        wsum = w.sum()
        if wsum <= 1e-9:
            return
        w = w / wsum
        P = _calc_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # 2-point: {0, 1} with weight p on 1
    for p in np.linspace(0.01, 0.99, 199):
        if time.time() > deadline:
            break
        s = np.array([0.0, 1.0])
        w = np.array([1-p, p])
        try_config(s, w)
    
    # 3-point: {0, t, 1} with weights
    # First, grid search over t
    for t in np.linspace(0.05, 0.95, 40):
        if time.time() > deadline:
            break
        s = np.array([0.0, t, 1.0])
        # Grid search over weights
        for w1 in np.linspace(0.0, 0.6, 20):
            for w2 in np.linspace(0.0, 0.6, 20):
                w0 = 1.0 - w1 - w2
                if w0 < -1e-9:
                    continue
                w = np.array([w0, w1, w2])
                try_config(s, w)
    
    # 4-point: {0, a, b, 1}
    for a in np.linspace(0.05, 0.4, 10):
        if time.time() > deadline:
            break
        for b in np.linspace(a+0.05, 0.95, 10):
            if time.time() > deadline:
                break
            s = np.array([0.0, a, b, 1.0])
            for _ in range(5):
                w = np.random.dirichlet(np.ones(4))
                try_config(s, w)
    
    # 5-point
    for _ in range(3):
        if time.time() > deadline:
            break
        s = np.array([0.0] + np.sort(np.random.uniform(0.05, 0.95, 3)) + [1.0])
        for _ in range(10):
            w = np.random.dirichlet(np.ones(5))
            try_config(s, w)
    
    # Refine best with local optimization
    if best_P > 0 and time.time() < deadline:
        n = len(best_support)
        if n <= 5:
            def objective(params):
                s = np.maximum(params[:n], 0.0)
                s = np.sort(s)
                w = np.maximum(params[n:], 0.0)
                wsum = w.sum()
                if wsum < 1e-9:
                    return 1e10
                w = w / wsum
                return -_calc_P(s, w)
            
            x0 = np.concatenate([best_support, best_weights])
            for _ in range(5):
                if time.time() > deadline:
                    break
                try:
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < -best_P:
                        s_opt = np.sort(np.maximum(res.x[:n], 0.0))
                        w_opt = np.maximum(res.x[n:], 0.0)
                        w_opt = w_opt / w_opt.sum()
                        try_config(s_opt, w_opt)
                        x0 = res.x
                except:
                    break
    
    return best_support, best_weights
