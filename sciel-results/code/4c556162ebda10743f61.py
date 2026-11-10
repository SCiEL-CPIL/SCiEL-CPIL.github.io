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
    
    # Strategy: The optimal distribution likely has mass concentrated at:
    # - Very small values (near 0) to minimize X1+X2+X3
    # - Large values to maximize 2*X4
    # Let's try various structural patterns
    
    # Phase 1: Try fixed support patterns with weight optimization
    # Pattern A: Two-point mass at 0 and large value
    # Pattern B: Three-point mass at 0, medium, large
    # Pattern C: More points with strategic placement
    
    # Try a few well-motivated initial structures
    structures = [
        # (support, weights) - initial guesses
        (np.array([0.0, 1.0]), np.array([0.7, 0.3])),
        (np.array([0.0, 0.5, 10.0]), np.array([0.5, 0.2, 0.3])),
        (np.array([0.0, 0.1, 0.5, 10.0]), np.array([0.4, 0.1, 0.1, 0.4])),
        (np.array([0.0, 0.01, 0.1, 1.0, 10.0]), np.array([0.3, 0.1, 0.1, 0.1, 0.4])),
    ]
    
    for s_init, w_init in structures:
        update_best(s_init, w_init)
    
    # Phase 2: Optimize weights for fixed support patterns
    for n_grid in [4, 8, 16]:
        if time_remaining() < 60:
            break
            
        # Create a support that emphasizes small and large values
        if n_grid == 4:
            s_init = np.array([0.0, 0.1, 1.0, 10.0])
        elif n_grid == 8:
            s_init = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
        else:
            # Log-spaced from 0 to 100, plus some small values
            s_init = np.concatenate([
                np.array([0.0, 0.001, 0.01, 0.1]),
                np.geomspace(1, 100, n_grid - 4)
            ])
        
        w_init = np.ones(n_grid) / n_grid
        
        def objective(w_flat, s_fixed=s_init):
            w = np.clip(w_flat, 0, 1)
            s_sum = np.sum(w)
            if s_sum < 1e-9:
                return 0.0
            w = w / s_sum
            return -calculate_P(s_fixed, w)
        
        bounds = [(0, 1)] * n_grid
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        try:
            res = minimize(objective, w_init, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-7})
            if res.success or res.fun < 0:
                update_best(s_init, res.x)
        except Exception:
            pass
    
    # Phase 3: Optimize both support and weights for small n
    def optimize_full(n_pts, s_init, w_init, n_restarts=3):
        if time_remaining() < 30:
            return
        
        def obj_full(params):
            s = params[:n_pts]
            w = params[n_pts:]
            
            s = np.clip(s, 0, 1e5)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        bounds_s = [(0, 1000)] * n_pts
        bounds_w = [(0, 1)] * n_pts
        bounds = bounds_s + bounds_w
        
        best_local = -np.inf
        best_params = None
        
        rng = np.random.default_rng(42 + n_pts)
        
        for restart in range(n_restarts):
            if time_remaining() < 15:
                break
            
            # Perturb initial guess
            s_pert = s_init * (1 + 0.2 * rng.standard_normal(n_pts))
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = w_init * (1 + 0.2 * rng.standard_normal(n_pts))
            w_pert = np.clip(w_pert, 0.01, 1)
            
            params_init = np.concatenate([s_pert, w_pert])
            
            try:
                res = minimize(obj_full, params_init, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_local:
                    best_local = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is not None:
            s_opt = best_params[:n_pts]
            w_opt = best_params[n_pts:]
            update_best(s_opt, w_opt)
    
    # Try different sizes with structurally motivated initial guesses
    if time_remaining() > 120:
        # n=2: two-point distribution
        optimize_full(2, np.array([0.0, 5.0]), np.array([0.6, 0.4]), n_restarts=5)
    
    if time_remaining() > 100:
        # n=3: three-point distribution
        optimize_full(3, np.array([0.0, 0.5, 10.0]), np.array([0.5, 0.1, 0.4]), n_restarts=5)
    
    if time_remaining() > 80:
        # n=4: four-point distribution
        optimize_full(4, np.array([0.0, 0.1, 1.0, 10.0]), np.array([0.4, 0.1, 0.1, 0.4]), n_restarts=4)
    
    if time_remaining() > 60:
        # n=5: five-point distribution
        optimize_full(5, np.array([0.0, 0.01, 0.1, 1.0, 10.0]), 
                     np.array([0.3, 0.1, 0.1, 0.1, 0.4]), n_restarts=3)
    
    # Phase 4: Refinement - polish the best solution found
    if best_support is not None and time_remaining() > 30:
        s_cur = best_support
        w_cur = best_weights
        n_pts = len(s_cur)
        
        def obj_polish(params):
            s = params[:n_pts]
            w = params[n_pts:]
            
            s = np.clip(s, 0, 1e5)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        # Tight bounds around current solution
        s_min = np.maximum(0, s_cur * 0.5)
        s_max = s_cur * 2.0 + 1
        w_min = np.maximum(0, w_cur - 0.1)
        w_max = np.minimum(1.0, w_cur + 0.1)
        
        bounds_polish = list(zip(s_min, s_max)) + list(zip(w_min, w_max))
        
        params_init = np.concatenate([s_cur, w_cur])
        
        try:
            res = minimize(obj_polish, params_init, method='SLSQP', bounds=bounds_polish,
                          options={'maxiter': 300, 'ftol': 1e-9})
            if res.fun < -best_P:
                update_best(res.x[:n_pts], res.x[n_pts:])
        except Exception:
            pass
        
        # Also try Nelder-Mead polish
        if time_remaining() > 15:
            try:
                res = minimize(obj_polish, params_init, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
                if res.fun < -best_P:
                    update_best(res.x[:n_pts], res.x[n_pts:])
            except Exception:
                pass
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 0.1, 1.0, 10.0])
        best_weights = np.array([0.4, 0.1, 0.1, 0.4])
    
    return best_support, best_weights
