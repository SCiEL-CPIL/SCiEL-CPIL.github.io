import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    For n <= 20, use the O(n^4) direct approach.
    For n > 20, use the O(n^2 * log(n^2)) sorted approach.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / np.sum(weights)
    
    if n <= 20:
        # Direct O(n^4) but with vectorization over i,j for each (k,l)
        # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
        # Precompute all pair sums and their weights
        S_12 = support[:, None] + support[None, :]  # n x n
        W_12 = w[:, None] * w[None, :]  # n x n
        
        P_total = 0.0
        for k in range(n):
            for l in range(n):
                thresh = 2 * support[l] - support[k]
                mask = S_12 < thresh
                prob_ij = np.sum(W_12 * mask)
                P_total += w[k] * w[l] * prob_ij
        return P_total
    else:
        # For larger n, sort support and use cumulative sums
        # Sort support and weights together
        idx = np.argsort(support)
        s_sorted = support[idx]
        w_sorted = w[idx]
        
        # Precompute pair sums sorted
        # For each k (X4), we need sum over l (X3) of w_l * P[X1+X2 < 2*s_l - s_k]
        # For each (k, l), we need sum_{i,j: s_i+s_j < thresh} w_i * w_j
        # where thresh = 2*s_l - s_k
        
        # Build all pair sums
        S_12 = s_sorted[:, None] + s_sorted[None, :]  # n x n
        S_12_flat = S_12.ravel()
        W_12_flat = (w_sorted[:, None] * w_sorted[None, :]).ravel()
        
        # Sort pair sums
        pair_idx = np.argsort(S_12_flat)
        S_12_sorted = S_12_flat[pair_idx]
        W_12_sorted = W_12_flat[pair_idx]
        W_12_cumsum = np.cumsum(W_12_sorted)
        
        P_total = 0.0
        for k in range(n):
            for l in range(n):
                thresh = 2 * s_sorted[l] - s_sorted[k]
                # Find count of pair sums < thresh
                pos = np.searchsorted(S_12_sorted, thresh, side='left')
                if pos == 0:
                    prob_ij = 0.0
                else:
                    prob_ij = W_12_cumsum[pos - 1]
                P_total += w_sorted[k] * w_sorted[l] * prob_ij
        return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if s is None or w is None:
            return
        try:
            # Validate
            if len(s) != len(w):
                return
            if len(s) == 0:
                return
            if np.any(np.isnan(s)) or np.any(np.isnan(w)):
                return
            if np.any(np.isinf(s)) or np.any(np.isinf(w)):
                return
            if np.any(s < -1e-9):
                return
            if np.any(w < -1e-9):
                return
            if np.sum(w) <= 1e-9:
                return
            
            p_val = _compute_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass
    
    # Strategy: Fixed fine grid with optimized weights
    # The key insight: if we fix the support to a fine grid, P becomes a quadratic form w^T M w
    # where M is precomputed. Then we can use efficient quadratic optimization.
    
    # Try different grid sizes and spacings
    grid_configs = [
        (32, 1.0, 32.0),   # linear 0..31
        (64, 1.0, 64.0),   # linear 0..63
        (32, 0.5, 16.0),   # finer spacing
        (48, 1.0, 48.0),
        (64, 0.5, 32.0),
        (96, 1.0, 96.0),
    ]
    
    for n_grid, spacing, max_val in grid_configs:
        if time.time() > deadline - 60:
            break
        
        support = np.linspace(0, max_val, n_grid, dtype=np.float64)
        
        # Precompute the matrix M such that P = w^T M w (with normalized w)
        # P = sum_{k,l,i,j} w_k w_l w_i w_j * I(s_i + s_j < 2*s_l - s_k)
        # This is a 4th order tensor contraction, not a simple quadratic form in w
        # But for fixed support, we can compute P(w) efficiently
        
        # Actually, let's just optimize weights directly
        def objective(w_flat):
            w = np.clip(w_flat, 0, 1)
            s_sum = np.sum(w)
            if s_sum < 1e-9:
                return 0.0
            w = w / s_sum
            return -_compute_P(support, w)
        
        # Multiple random initializations
        rng = np.random.default_rng(42 + n_grid)
        
        best_local = None
        best_local_val = np.inf
        
        for restart in range(3):
            if time.time() > deadline - 60:
                break
            
            if restart == 0:
                w_init = np.ones(n_grid) / n_grid
            elif restart == 1:
                # Concentrate on low values
                w_init = np.exp(-np.linspace(0, 2, n_grid))
                w_init = w_init / np.sum(w_init)
            else:
                # Random
                w_init = rng.random(n_grid)
                w_init = w_init / np.sum(w_init)
            
            try:
                res = minimize(objective, w_init, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-8})
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local = res.x
            except Exception:
                pass
        
        if best_local is not None:
            update_best(support, best_local)
    
    # Also try small support optimization with both positions and weights
    small_configs = [
        (3, np.array([0.0, 1.0, 5.0])),
        (4, np.array([0.0, 0.5, 2.0, 8.0])),
        (5, np.array([0.0, 0.5, 1.5, 4.0, 10.0])),
    ]
    
    for n_pts, s_init in small_configs:
        if time.time() > deadline - 30:
            break
        
        w_init = np.ones(n_pts) / n_pts
        
        def obj_small(params):
            s = params[:n_pts]
            w = params[n_pts:]
            s = np.clip(s, 0, 1000)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            return -_compute_P(s, w)
        
        bounds = [(0, 1000)] * n_pts + [(0, 1)] * n_pts
        
        best_local = None
        best_local_val = np.inf
        
        rng = np.random.default_rng(100 + n_pts)
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            s_pert = s_init * (1 + 0.2 * rng.standard_normal(n_pts))
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = w_init * (1 + 0.2 * rng.standard_normal(n_pts))
            w_pert = np.clip(w_pert, 0.01, 1)
            
            params_init = np.concatenate([s_pert, w_pert])
            
            try:
                res = minimize(obj_small, params_init, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local = res.x
            except Exception:
                pass
        
        if best_local is not None:
            s_opt = best_local[:n_pts]
            w_opt = best_local[n_pts:]
            update_best(s_opt, w_opt)
    
    # Fallback
    if best_support is None:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
    
    return best_support, best_weights
