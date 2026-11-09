# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,), assumed normalized
    """
    s = support
    w = weights
    n = len(s)
    
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
            # Find number of pairs with sum < threshold
            idx = np.searchsorted(sums_sorted, threshold, side='left')
            inner_sum = cumsum_w[idx - 1] if idx > 0 else 0.0
            P += w[k] * w[l] * inner_sum
            
    return P

def _optimize_both(support_init, weights_init, n, deadline):
    """
    Optimize both support and weights jointly.
    Parameters: support_init, weights_init - initial values
    Returns: (best_P, best_support, best_weights)
    """
    def objective(params):
        if time.time() > deadline:
            return 1e6
        s = params[:n]
        w = params[n:]
        
        # Ensure support is non-negative
        s = np.maximum(s, 0.0)
        # Ensure weights are non-negative
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return 1e6
        w = w / w_sum
        
        return -_calc_P(s, w)
    
    # Flatten initial params
    x0 = np.concatenate([support_init, weights_init])
    
    try:
        res = so.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.success:
            s_opt = np.maximum(res.x[:n], 0.0)
            w_opt = np.maximum(res.x[n:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum > 1e-9:
                w_opt = w_opt / w_sum
                P_val = _calc_P(s_opt, w_opt)
                return P_val, s_opt, w_opt
    except Exception:
        pass
    
    return 0.0, support_init, weights_init

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 585.0  # 15s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0])
    best_weights = np.array([1.0])
    
    # Strategy 1: Optimize weights for fixed supports (fast, many restarts)
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Try multiple support configurations
        support_configs = [
            np.arange(n, dtype=np.float64),
            np.linspace(0, 1, n),
            np.linspace(0, 10, n),
            np.array([0.0] + np.random.uniform(0.1, 5.0, n-1)),
        ]
        
        for support in support_configs:
            if time.time() > deadline:
                break
            
            # Sort support
            support = np.sort(support)
            
            # Multiple random weight initializations
            for _ in range(10):
                if time.time() > deadline:
                    break
                    
                w_init = np.random.dirichlet(np.ones(n))
                
                def objective_w(w):
                    if np.any(w < -1e-9):
                        return 1e6
                    w_norm = w / np.sum(w)
                    return -_calc_P(support, w_norm)
                
                def constraint_sum(w):
                    return np.sum(w) - 1.0
                
                constraints = [{'type': 'eq', 'fun': constraint_sum}]
                bounds = [(0, 1)] * n
                
                try:
                    res = so.minimize(
                        objective_w, 
                        w_init, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints,
                        options={'maxiter': 50, 'ftol': 1e-7}
                    )
                    if res.success:
                        w_opt = np.clip(res.x, 0, None)
                        w_opt = w_opt / np.sum(w_opt)
                        P_val = _calc_P(support, w_opt)
                        
                        if P_val > best_P:
                            best_P = P_val
                            best_support = support.copy()
                            best_weights = w_opt.copy()
                            
                except Exception:
                    pass
    
    # Strategy 2: Joint optimization of support and weights
    for n in [2, 3, 4]:
        if time.time() > deadline:
            break
            
        for _ in range(15):
            if time.time() > deadline:
                break
                
            # Random initial support (sorted, non-negative)
            s_init = np.sort(np.concatenate([np.array([0.0]), np.random.uniform(0.1, 5.0, n-1)]))
            w_init = np.random.dirichlet(np.ones(n))
            
            P_val, s_opt, w_opt = _optimize_both(s_init, w_init, n, deadline)
            
            if P_val > best_P:
                best_P = P_val
                best_support = s_opt
                best_weights = w_opt
    
    # Strategy 3: Try specific analytical structures
    # 2-point: {0, M} with prob p on M
    for M in [0.5, 1.0, 2.0, 5.0, 10.0]:
        if time.time() > deadline:
            break
        for p in np.linspace(0.1, 0.9, 17):
            s = np.array([0.0, M])
            w = np.array([1-p, p])
            P_val = _calc_P(s, w)
            if P_val > best_P:
                best_P = P_val
                best_support = s.copy()
                best_weights = w.copy()
    
    # If no optimization succeeded, fall back to seed
    if best_P == 0.0:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
