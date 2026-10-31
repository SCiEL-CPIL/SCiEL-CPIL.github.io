"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution."""
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def _ratio_normalized(h):
    """Compute ratio with L1 norm fixed to 1. Returns ||h||_2^2 / ||h*h||_inf."""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    max_conv = _autoconv_max(h_norm)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / max_conv

def _is_sidon(support):
    """Check if a set is a Sidon set (all pairwise sums distinct)."""
    s = sorted(support)
    sums = {}
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                if sums[total] != (i, j):
                    return False
            else:
                sums[total] = (i, j)
    return True

def _find_maximal_sidon(n, seed=42, max_size=12):
    """Find a maximal Sidon set within [0, n) using greedy + local search."""
    rng = np.random.default_rng(seed)
    
    # Greedy construction
    support = set()
    candidates = list(range(n))
    rng.shuffle(candidates)
    
    for idx in candidates:
        if len(support) >= max_size:
            break
        new_support = support | {idx}
        if _is_sidon(new_support):
            support.add(idx)
    
    # Local search: try to add more elements or swap to improve
    improved = True
    while improved and len(support) < max_size:
        improved = False
        for idx in range(n):
            if idx in support:
                continue
            new_support = support | {idx}
            if _is_sidon(new_support):
                support.add(idx)
                improved = True
                break
    
    return sorted(support)

def _optimize_weights_iterative(support, max_iter=1000, tol=1e-10):
    """Optimize weights on a fixed support using iterative projection method."""
    n = len(support)
    if n == 0:
        return np.array([])
    
    # Initialize with equal weights
    weights = np.ones(n) / n
    
    for _ in range(max_iter):
        h = np.zeros(N)
        h[support] = weights
        val_before = _ratio_normalized(h)
        
        # Gradient ascent step on log-weights (to maintain positivity)
        h_norm = h / np.sum(h)
        l2_sq = np.sum(h_norm**2)
        c = np.correlate(h_norm, h_norm, mode='full')
        max_conv = np.max(c)
        
        # Compute gradient of ratio w.r.t. weights
        # ratio = l2_sq / max_conv
        # d(ratio)/d(w_i) = 2*h_i*l1^-2*max_conv^-1 - l2_sq*l1^-2*max_conv^-2 * d(max_conv)/d(w_i)
        # Simplified: use numerical gradient
        grad = np.zeros(n)
        eps = 1e-6
        for i in range(n):
            w_plus = weights.copy()
            w_plus[i] += eps
            w_plus = np.maximum(w_plus, 0)
            w_plus /= np.sum(w_plus)
            h_plus = np.zeros(N)
            h_plus[support] = w_plus
            val_plus = _ratio_normalized(h_plus)
            grad[i] = (val_plus - val_before) / eps
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Take a step in gradient direction
        step_size = 0.01
        new_weights = weights + step_size * grad / grad_norm
        new_weights = np.maximum(new_weights, 0)
        new_sum = np.sum(new_weights)
        if new_sum < 1e-15:
            break
        new_weights /= new_sum
        
        # Check convergence
        h_new = np.zeros(N)
        h_new[support] = new_weights
        val_after = _ratio_normalized(h_new)
        
        if abs(val_after - val_before) < tol:
            weights = new_weights
            break
        
        weights = new_weights
    
    return weights

def _local_search_sidon(support, max_iters=500, deadline=None):
    """Local search on Sidon set structure: swap elements to minimize max convolution."""
    support = list(support)
    n = len(support)
    
    if n == 0:
        return support
    
    # Get initial weights
    weights = _optimize_weights_iterative(support, max_iter=200)
    h = np.zeros(N)
    h[support] = weights
    best_val = _ratio_normalized(h)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try swapping each element with each non-support element
        for i in range(n):
            for j in range(N):
                if j in support:
                    continue
                
                new_support = support.copy()
                new_support[i] = j
                new_support = sorted(new_support)
                
                # Check if still Sidon
                if not _is_sidon(new_support):
                    continue
                
                # Optimize weights on new support
                new_weights = _optimize_weights_iterative(new_support, max_iter=100)
                new_h = np.zeros(N)
                new_h[new_support] = new_weights
                new_val = _ratio_normalized(new_h)
                
                if new_val > best_val + 1e-8:
                    support = new_support
                    weights = new_weights
                    best_val = new_val
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return support

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio_normalized(best_h)
    
    # Try various support sizes with Sidon structures
    for support_size in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random seeds for diversity
        for seed in [42, 123, 777, 2024, 555, 999]:
            if time.time() > deadline - 10:
                break
            
            # Find a Sidon set
            support = _find_maximal_sidon(N, seed=seed, max_size=support_size)
            if len(support) < 2:
                continue
            
            # Local search to improve the support
            support = _local_search_sidon(support, max_iters=100, deadline=deadline - 5)
            
            # Optimize weights
            weights = _optimize_weights_iterative(support, max_iter=500)
            
            h = np.zeros(N)
            h[support] = weights
            
            val = _ratio_normalized(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    # Final refinement: local search from best configuration
    if time.time() < deadline - 5:
        # Extract support from best_h
        support = [i for i in range(N) if best_h[i] > 1e-10]
        
        if len(support) >= 2:
            # More extensive local search
            support = _local_search_sidon(support, max_iters=300, deadline=deadline - 2)
            
            # Optimize weights
            weights = _optimize_weights_iterative(support, max_iter=1000)
            
            h = np.zeros(N)
            h[support] = weights
            
            val = _ratio_normalized(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
