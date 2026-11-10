import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s buffer for safety
    
    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
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
        
        # Vectorized calculation for X3 and X4
        # We need sum over k, l of p_k * p_l * P(Z < 2*s_l - s_k)
        
        # Create meshgrid for s_k and s_l
        # s_k shape (n, 1), s_l shape (1, n) -> thresholds shape (n, n)
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # For each threshold, find idx = number of s_sorted < threshold
        # np.searchsorted returns the index of the first element >= threshold
        # So the number of elements < threshold is exactly this index.
        # Note: s_sorted is 1D. thresholds is 2D.
        # searchsorted accepts 1D array and 2D array? 
        # Yes, if we broadcast or use advanced indexing.
        # Actually, np.searchsorted works element-wise if the second arg is array-like.
        # Let's check if it broadcasts. 
        # np.searchsorted(a, v) where a is 1D and v is 2D returns 2D array of indices.
        
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        # Now we need p_prefix[idxs - 1] where idxs > 0, else 0
        # Use np.take with mode='clip' to handle idx=0 (returns p_prefix[0] which is p_sorted[0], but we want 0)
        # Better: 
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        # For idxs > 0, we want p_prefix[idxs - 1]
        # We can use advanced indexing
        if np.any(mask):
            # idxs[mask] are the indices. We need p_prefix[idxs[mask] - 1]
            # p_prefix has length n^2. idxs are in [0, n^2].
            # If idxs == n^2, idxs-1 is n^2-1, valid.
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        # Total probability
        # P = sum_{k,l} p_k * p_l * prob_z_less[k,l]
        # p_k is p[:, None], p_l is p[None, :]
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Try different sizes n
    # Smaller n is faster per eval, allowing more restarts/iterations
    n_sizes = [10, 15, 20]
    
    rng = np.random.default_rng(42)
    
    for n in n_sizes:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on n
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            
            # Initialize support: Random sorted points in [0, 1] or [0, 2]
            # Often optimal supports are within [0, 1] or close
            # Strategy: Mix of small values (for X1,X2,X3) and large values (for X4)
            # Let's try a structured initialization:
            # Some points near 0, some spread out.
            
            # Random uniform
            support_init = np.sort(rng.uniform(0, 1, n))
            weights_init = rng.dirichlet(np.ones(n))
            
            current_support = support_init.copy()
            current_weights = weights_init.copy()
            current_score = compute_score(current_support, current_weights)
            
            # Local Search: Random Perturbation (Hill Climbing)
            # We perturb support and weights
            max_iters = 500  # Increased due to faster score calc
            step_size_s = 0.05
            step_size_w = 0.05
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Perturb support
                idx = rng.integers(n)
                delta_s = rng.normal(0, step_size_s)
                new_support = current_support.copy()
                new_support[idx] += delta_s
                if new_support[idx] < 0:
                    new_support[idx] = 0
                
                # Perturb weights
                new_weights = current_weights.copy()
                new_weights += rng.normal(0, step_size_w, n)
                new_weights = np.maximum(new_weights, 0)
                if np.sum(new_weights) < 1e-9:
                    continue
                new_weights = new_weights / np.sum(new_weights)
                
                new_score = compute_score(new_support, new_weights)
                
                if new_score > current_score:
                    current_support = new_support
                    current_weights = new_weights
                    current_score = new_score
                    # Adaptive step size: reduce if we move often? 
                    # Or increase if stuck? 
                    # Simple: keep constant for now, maybe decay slowly
                    # step_size_s *= 0.99
                    # step_size_w *= 0.99
                else:
                    # If stuck, maybe increase step size occasionally?
                    if it % 50 == 0:
                        step_size_s *= 1.1
                        step_size_w *= 1.1
                        step_size_s = min(step_size_s, 0.5)
                        step_size_w = min(step_size_w, 0.5)

            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()
                best_weights = current_weights.copy()

    if best_support is None:
        # Fallback to seed
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n

    # Clean up: ensure non-negative
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    return best_support, best_weights
