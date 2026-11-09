import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 40s buffer for safety
    
    def compute_score(support, weights):
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights
        # s_i + s_j
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z = X1 + X2
        p_prefix = np.cumsum(p_sorted)
        
        total_prob = 0.0
        
        # Iterate over k (X3) and l (X4)
        for k in range(n):
            sk = support[k]
            pk = p[k]
            for l in range(n):
                sl = support[l]
                pl = p[l]
                # Threshold T = 2 * s_l - s_k
                T = 2.0 * sl - sk
                # Find number of pairs (i, j) such that s_i + s_j < T
                # Strict inequality: s_i + s_j < T
                # np.searchsorted with side='left' gives index of first element >= T
                # We want elements strictly less than T, so index is the number of elements < T
                idx = np.searchsorted(s_sorted, T, side='left')
                
                if idx == 0:
                    prob_z_less = 0.0
                else:
                    prob_z_less = p_prefix[idx - 1]
                    
                total_prob += pk * pl * prob_z_less
                
        return total_prob

    def compute_score_vectorized(support, weights):
        """Vectorized score computation for faster evaluation."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums
        p_prefix = np.cumsum(p_sorted)
        
        # Build thresholds matrix: T[k, l] = 2*s_l - s_k
        # For each (k, l), we need P(Z < T[k, l]) where Z = X1+X2
        # T is n x n matrix
        T_matrix = 2.0 * support[None, :] - support[:, None]  # T[k, l] = 2*s_l - s_k
        
        # For each threshold, find index via searchsorted
        # We need to do this for each of the n^2 thresholds
        # This is slow if done one by one. Let's try a different approach.
        
        # Alternative: For each pair (i,j) in the sum, and each (k,l), check if s_i+s_j < 2*s_l - s_k
        # This is O(n^4) which is too slow.
        
        # Let's stick with the loop but optimize it slightly
        total_prob = 0.0
        for k in range(n):
            sk = support[k]
            pk = p[k]
            for l in range(n):
                sl = support[l]
                pl = p[l]
                T = 2.0 * sl - sk
                idx = np.searchsorted(s_sorted, T, side='left')
                if idx > 0:
                    total_prob += pk * pl * p_prefix[idx - 1]
        return total_prob

    best_score = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(123)
    
    # Try different sizes
    # Smaller n allows more iterations. n=8, 10, 12 are good.
    n_sizes = [8, 10, 12]
    
    for n in n_sizes:
        if time.time() > deadline:
            break
        
        # Multiple restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Strategy: Place points in [0, 1]
            # Often optimal solutions have a cluster near 0 and a few larger points
            # Let's try a mix of uniform and structured init
            
            if r == 0:
                # Uniform
                support_init = np.sort(rng.uniform(0, 1, n))
            elif r == 1:
                # Log-spaced
                support_init = np.sort(rng.logspace(-3, 0, n))
                support_init = support_init / support_init.max()
            elif r == 2:
                # Clustered near 0
                support_init = np.sort(rng.exponential(0.1, n))
                support_init = support_init / (support_init.max() + 1e-9)
            else:
                # Random
                support_init = np.sort(rng.uniform(0, 1, n))
            
            weights_init = rng.dirichlet(np.ones(n) * 0.5)
            
            current_support = support_init.copy()
            current_weights = weights_init.copy()
            current_score = compute_score(current_support, current_weights)
            
            # Local optimization using Nelder-Mead
            # We need to optimize 2n parameters: n support points and n weights
            # But weights sum to 1, so n-1 free parameters
            # Support points must be non-negative
            
            def objective(params):
                # params: [s0, s1, ..., s_{n-1}, w0, w1, ..., w_{n-2}]
                # w_{n-1} = 1 - sum(w0..w_{n-2})
                s = params[:n]
                w = params[n:2*n-1]
                
                # Ensure non-negative support
                s = np.maximum(s, 0)
                
                # Ensure positive weights
                w = np.maximum(w, 1e-8)
                w_last = 1.0 - np.sum(w)
                if w_last < 1e-8:
                    return -1.0  # Invalid
                w_full = np.append(w, w_last)
                
                score = compute_score(s, w_full)
                return -score  # Minimize negative score
            
            # Initial params
            s_init = current_support.copy()
            w_init = current_weights.copy()
            w_params = w_init[:n-1]
            
            params_init = np.concatenate([s_init, w_params])
            
            # Optimize
            try:
                res = minimize(objective, params_init, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
                
                if res.success or res.fun < objective(params_init):
                    opt_params = res.x
                    opt_s = np.maximum(opt_params[:n], 0)
                    opt_w = opt_params[n:2*n-1]
                    opt_w = np.maximum(opt_w, 1e-8)
                    opt_w_last = 1.0 - np.sum(opt_w)
                    if opt_w_last >= 1e-8:
                        opt_w_full = np.append(opt_w, opt_w_last)
                        opt_score = -res.fun
                        if opt_score > current_score:
                            current_support = opt_s
                            current_weights = opt_w_full
                            current_score = opt_score
            except Exception:
                pass
            
            # Additional local search: random perturbations
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Perturb support
                idx = rng.integers(n)
                delta_s = rng.normal(0, 0.02)
                new_support = current_support.copy()
                new_support[idx] = max(0, new_support[idx] + delta_s)
                
                # Perturb weights
                new_weights = current_weights.copy()
                new_weights += rng.normal(0, 0.02, n)
                new_weights = np.maximum(new_weights, 0)
                w_sum = np.sum(new_weights)
                if w_sum < 1e-9:
                    continue
                new_weights = new_weights / w_sum
                
                new_score = compute_score(new_support, new_weights)
                
                if new_score > current_score:
                    current_support = new_support
                    current_weights = new_weights
                    current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()
                best_weights = current_weights.copy()
                
                # Check if we've reached a very high score
                if best_score > 0.95:
                    # Try to refine further
                    pass
    
    if best_support is None:
        # Fallback
        n = 8
        support = np.sort(rng.uniform(0, 1, n))
        weights = np.ones(n) / n
        return support, weights
    
    # Clean up
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    
    return best_support, best_weights
