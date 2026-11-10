import numpy as np
import time
from scipy.optimize import minimize

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4]
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / np.sum(weights)
    
    # Precompute pairwise sums for X1+X2 (size n x n)
    S_12 = support[:, None] + support[None, :]
    W_12 = w[:, None] * w[None, :]
    
    P_total = 0.0
    for k in range(n):
        target = 2 * support[k]
        prob_k = 0.0
        for l in range(n):
            thresh = target - support[l]
            mask = S_12 < thresh
            prob_l = np.sum(W_12 * mask)
            prob_k += w[l] * prob_l
        P_total += w[k] * prob_k
        
    return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 580.0
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if s is None or w is None:
            return
        try:
            s = np.asarray(s, dtype=np.float64).flatten()
            w = np.asarray(w, dtype=np.float64).flatten()
            if len(s) != len(w) or len(s) == 0:
                return
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                return
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                return
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return
            if np.sum(w) <= 1e-9:
                return
            
            p_val = calculate_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass

    def time_remaining():
        return deadline - time.time()

    # Structural Optimization for Small N
    # We suspect the optimal distribution has a "bimodal" or "trimodal" structure
    # with mass at 0, some small values, and a large value M.
    
    def optimize_structure(n_pts, s_init, w_init, max_iter=1000):
        if time_remaining() < 10:
            return
        
        # Parameters: support points (positive) and weights (positive, sum to 1)
        # To ensure positivity and sum-to-1, we can optimize:
        # s_i > 0, w_i > 0, sum(w)=1
        # SLSQP can handle this directly with constraints and bounds.
        
        def objective(params):
            s = params[:n_pts]
            w = params[n_pts:]
            
            # Clip to valid ranges to help optimizer
            s = np.clip(s, 1e-6, 1000)
            w = np.clip(w, 1e-6, 1)
            
            # Normalize weights
            w_sum = np.sum(w)
            if w_sum <= 1e-9:
                return 0.0
            w = w / w_sum
            
            # Sort support to maintain structure? 
            # The optimizer might permute points. 
            # It's better to fix the order: 0 = s_0 < s_1 < ... < s_{n-1} = M.
            # So we enforce s[0]=0 and s is increasing.
            # Let's re-parameterize to enforce order.
            # s[0] = 0
            # s[i] = s[i-1] + exp(delta_i) for i > 0
            # This is complex for SLSQP.
            # Alternative: Just optimize s and w, and sort s and corresponding w at the end?
            # No, the objective function is invariant to permutation of (s_i, w_i) pairs.
            # So we can just optimize s, w with s > 0, w > 0.
            
            return -calculate_P(s, w)
        
        # Initial guess
        s0 = np.array(s_init, dtype=float)
        w0 = np.array(w_init, dtype=float)
        w0 = w0 / np.sum(w0)
        
        x0 = np.concatenate([s0, w0])
        
        # Bounds
        bounds_s = [(1e-6, 1000)] * n_pts
        bounds_w = [(1e-6, 1)] * n_pts
        bounds = bounds_s + bounds_w
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x[n_pts:]) - 1.0})
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-8})
            
            if res.success or res.fun < 0:
                s_opt = res.x[:n_pts]
                w_opt = res.x[n_pts:]
                
                # Ensure s is non-negative
                s_opt = np.clip(s_opt, 0, None)
                w_opt = np.clip(w_opt, 0, None)
                
                # Sort by support value to canonicalize
                # This helps if the optimizer swapped points
                idx = np.argsort(s_opt)
                s_opt = s_opt[idx]
                w_opt = w_opt[idx]
                
                update_best(s_opt, w_opt)
                
        except Exception:
            pass

    # Define candidate structures
    # Structure 1: {0, M}
    # Optimal p for weight of 0 was ~0.58.
    # So weight of M is ~0.42.
    # Let's try M=1, M=10, M=100.
    for M in [1.0, 10.0, 100.0]:
        s_init = np.array([0.0, M])
        w_init = np.array([0.58, 0.42])
        optimize_structure(2, s_init, w_init)

    # Structure 2: {0, a, M}
    # Try various ratios a/M
    for M in [10.0, 100.0]:
        for a_ratio in [0.01, 0.1, 0.5]:
            a = a_ratio * M
            s_init = np.array([0.0, a, M])
            # Initial weights: more mass on 0 and M
            w_init = np.array([0.5, 0.1, 0.4])
            optimize_structure(3, s_init, w_init)

    # Structure 3: {0, a1, a2, M}
    for M in [100.0]:
        s_init = np.array([0.0, 0.1, 1.0, M])
        w_init = np.array([0.4, 0.1, 0.1, 0.4])
        optimize_structure(4, s_init, w_init)
        
        s_init = np.array([0.0, 0.01, 0.5, M])
        w_init = np.array([0.4, 0.1, 0.1, 0.4])
        optimize_structure(4, s_init, w_init)

    # Structure 4: {0, a1, a2, a3, M}
    if time_remaining() > 60:
        M = 100.0
        s_init = np.array([0.0, 0.01, 0.1, 1.0, M])
        w_init = np.array([0.3, 0.1, 0.1, 0.1, 0.4])
        optimize_structure(5, s_init, w_init)

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.58, 0.42])
    
    return best_support, best_weights
