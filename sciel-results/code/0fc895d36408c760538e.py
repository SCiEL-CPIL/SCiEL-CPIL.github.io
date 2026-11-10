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
    deadline = start_time + 600.0  # Safety margin: 630s limit, use 600s for computation
    
    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
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
        
        # Vectorized computation over all (k, l) pairs
        # Thresholds: T[k,l] = 2*s_l - s_k
        # For each (k,l), find idx = searchsorted(s_sorted, T[k,l], side='left')
        # prob_z_less[k,l] = p_prefix[idx-1] if idx > 0 else 0
        # total = sum_{k,l} p_k * p_l * prob_z_less[k,l]
        
        # Build threshold matrix: shape (n, n)
        # T[k, l] = 2 * support[l] - support[k]
        T = 2.0 * support[None, :] - support[:, None]  # shape (n_k, n_l)
        
        # Use searchsorted on the sorted array for each threshold
        # np.searchsorted can handle 2D input if we flatten and reshape
        T_flat = T.flatten()
        idx_flat = np.searchsorted(s_sorted, T_flat, side='left')
        idx = idx_flat.reshape(n, n)
        
        # For idx=0, prob is 0; for idx>0, prob is p_prefix[idx-1]
        # We need to be careful with indexing
        safe_idx = np.maximum(idx - 1, 0)
        prob_z_less = p_prefix[safe_idx]
        # Where idx was 0, set to 0
        prob_z_less = np.where(idx > 0, prob_z_less, 0.0)
        
        # Weighted sum: sum_{k,l} p_k * p_l * prob_z_less[k,l]
        p_k = p[:, None]  # shape (n, 1)
        p_l = p[None, :]  # shape (1, n)
        total_prob = np.sum(p_k * p_l * prob_z_less)
        
        return total_prob

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Try different sizes n
    n_sizes = [8, 12, 16]
    
    rng = np.random.default_rng(42)
    
    for n in n_sizes:
        if time.time() > deadline:
            break
            
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            
            # Initialize support: Random sorted points in [0, 1]
            support_init = np.sort(rng.uniform(0, 1, n))
            weights_init = rng.dirichlet(np.ones(n))
            
            current_support = support_init.copy()
            current_weights = weights_init.copy()
            current_score = compute_score(current_support, current_weights)
            
            # Local Search with adaptive step size
            max_iters = 200
            step_size = 0.1  # Start with larger steps
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Adaptive step size: decrease as we get closer to convergence
                effective_step = step_size * (1.0 - 0.5 * it / max_iters)
                
                # Perturb support: move one or two points
                n_perturb = rng.integers(1, 3)
                indices = rng.choice(n, size=n_perturb, replace=False)
                new_support = current_support.copy()
                for idx in indices:
                    delta_s = rng.normal(0, effective_step)
                    new_support[idx] += delta_s
                new_support = np.maximum(new_support, 0)
                
                # Perturb weights
                new_weights = current_weights.copy()
                new_weights += rng.normal(0, effective_step * 0.5, n)
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
                    # Reduce step size on success
                    step_size *= 0.95
                else:
                    # Occasionally increase step size to escape local minima
                    if rng.random() < 0.05:
                        step_size = min(step_size * 1.2, 0.2)
            
            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()
                best_weights = current_weights.copy()
                
    # Final polish: run a more careful local search on the best solution
    if best_support is not None and time.time() < deadline - 30:
        current_support = best_support.copy()
        current_weights = best_weights.copy()
        current_score = best_score
        n = len(current_support)
        
        # Polish with smaller steps and more iterations
        step_size = 0.02
        max_polish_iters = 500
        
        for it in range(max_polish_iters):
            if time.time() > deadline - 10:
                break
            
            # Perturb one support point
            idx = rng.integers(n)
            delta_s = rng.normal(0, step_size)
            new_support = current_support.copy()
            new_support[idx] += delta_s
            new_support = np.maximum(new_support, 0)
            
            # Perturb weights slightly
            new_weights = current_weights.copy()
            new_weights += rng.normal(0, step_size * 0.3, n)
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
                step_size *= 0.98
            else:
                if rng.random() < 0.02:
                    step_size = min(step_size * 1.1, 0.05)
        
        if current_score > best_score:
            best_score = current_score
            best_support = current_support
            best_weights = current_weights
    
    if best_support is None:
        # Fallback
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n

    # Clean up: ensure non-negative
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    return best_support, best_weights
