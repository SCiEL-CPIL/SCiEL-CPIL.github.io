import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import itertools

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
    
    P_total = 0.0
    for k in range(n):
        target = 2 * support[k]
        prob_k = 0.0
        for l in range(n):
            thresh = target - support[l]
            mask = S_12 < thresh
            prob_l = np.sum(w[:, None] * w[None, :] * mask)
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
            p_val = calculate_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass

    # 1. Try the known good uniform grid with weight optimization
    for n_grid in [16, 32, 64]:
        s_init = np.arange(n_grid, dtype=np.float64)
        w_init = np.ones(n_grid) / n_grid
        
        def objective(w_flat):
            w = np.clip(w_flat, 0, 1)
            s_sum = np.sum(w)
            if s_sum < 1e-9:
                return 0.0
            w = w / s_sum
            return -calculate_P(s_init, w)
        
        bounds = [(0, 1)] * n_grid
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        try:
            res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                update_best(s_init, res.x)
        except Exception:
            pass
        
        if time.time() > deadline:
            return best_support, best_weights

    # 2. Try small support sizes with optimized positions AND weights
    # This is where the "structural" improvement comes in: optimizing the support itself.
    
    def optimize_small_support(n_pts):
        # Initial guess: log-spaced or linear
        if n_pts == 2:
            s_init = np.array([0.0, 10.0])
            w_init = np.array([0.5, 0.5])
        elif n_pts == 3:
            s_init = np.array([0.0, 1.0, 10.0])
            w_init = np.array([1/3, 1/3, 1/3])
        elif n_pts == 4:
            s_init = np.array([0.0, 0.5, 2.0, 10.0])
            w_init = np.ones(4)/4
        else:
            return
        
        def obj_small(params):
            # First n_pts are supports, next n_pts are weights
            s = params[:n_pts]
            w = params[n_pts:]
            
            # Enforce non-negative supports (sort them to keep order, though order doesn't matter for P)
            # Actually, support order doesn't matter for the sum, but we need to handle duplicates or negatives.
            # Clip supports to be non-negative.
            s = np.clip(s, 0, 1e6)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            # Ensure supports are distinct enough? No, duplicates are fine, they just merge effectively.
            return -calculate_P(s, w)
        
        bounds_s = [(0, 1000)] * n_pts
        bounds_w = [(0, 1)] * n_pts
        bounds = bounds_s + bounds_w
        
        # Start from a few random-ish initializations
        best_local_p = -1.0
        best_local_params = None
        
        rng = np.random.default_rng(42)
        
        for restart in range(5):
            # Perturb initial guess
            s_pert = s_init * (1 + 0.1 * rng.standard_normal(n_pts))
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = w_init * (1 + 0.1 * rng.standard_normal(n_pts))
            w_pert = np.clip(w_pert, 0.01, 1)
            
            params_init = np.concatenate([s_pert, w_pert])
            
            try:
                res = minimize(obj_small, params_init, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_local_p:
                    best_local_p = res.fun
                    best_local_params = res.x
            except Exception:
                pass
            
            if time.time() > deadline:
                break
        
        if best_local_params is not None:
            s_opt = best_local_params[:n_pts]
            w_opt = best_local_params[n_pts:]
            update_best(s_opt, w_opt)

    optimize_small_support(2)
    optimize_small_support(3)
    optimize_small_support(4)
    optimize_small_support(5)
    
    if best_support is None:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
        
    return best_support, best_weights
