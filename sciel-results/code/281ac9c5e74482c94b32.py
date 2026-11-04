"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import numpy as np
import time

N = 50

def _compute_ratio(heights):
    """Compute the objective: (L2 norm)^2 / (L1 norm * sup norm of autoconvolution).
    
    heights: array of N nonnegative values
    Returns the ratio value.
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Autoconvolution
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(autoconv)
    
    if sup_norm <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)

def _coordinate_descent(h_init, deadline, max_iters=100):
    """Optimize using coordinate descent with binary search on each coordinate."""
    h = np.maximum(np.asarray(h_init, dtype=np.float64), 0.0).copy()
    current_score = _compute_ratio(h)
    best_h = h.copy()
    best_score = current_score
    
    # Determine a reasonable upper bound for any single height
    # If one height is very large, it dominates L2 and L1, and sup of autoconv.
    # Ratio approx h_i^2 / (h_i * h_i^2) = 1/h_i. So large heights are bad.
    # However, we need a bound for search. 
    # Let's use the current max or a multiple of the mean.
    h_max = max(np.max(h) * 10.0, 1.0)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle indices for stochastic coordinate descent
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            h_old = h[i]
            
            # Binary search for optimal h[i] in [0, h_max]
            # The objective is unimodal in h[i] for fixed others? 
            # Likely, but let's use a robust grid + refine approach or ternary search.
            # Ternary search is efficient for unimodal functions.
            lo = 0.0
            hi = h_max
            
            # Check if current is already better than bounds?
            # Just do ternary search
            for _ in range(30): # 30 steps gives high precision
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                h[i] = m1
                score_m1 = _compute_ratio(h)
                
                h[i] = m2
                score_m2 = _compute_ratio(h)
                
                if score_m1 > score_m2:
                    hi = m2
                else:
                    lo = m1
            
            h[i] = (lo + hi) / 2.0
            new_score = _compute_ratio(h)
            
            if new_score > best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            
            # If we didn't improve and are close to old, break early? 
            # Keep simple: continue.
            
        if not improved:
            break
            
    return best_h, best_score

def construct():
    """Construct 50 nonnegative heights to maximize the ratio.
    
    Returns a list of N nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with uniform (seed)
    best_heights = np.ones(N)
    best_score = _compute_ratio(best_heights)
    
    # Multiple random restarts
    rng = np.random.RandomState(42)
    
    # We will run several restarts with different initializations
    # Coordinate descent is fast, so we can do many.
    
    # Strategy:
    # 1. Random sparse vectors
    # 2. Random smooth vectors
    # 3. Perturbed uniform
    
    for trial in range(50): # Max 50 restarts, will break on deadline
        if time.time() > deadline - 5.0:
            break
            
        # Generate initialization
        if trial % 3 == 0:
            # Sparse: random few non-zeros
            h_init = np.zeros(N)
            num_nonzero = rng.randint(1, 10)
            indices = rng.choice(N, num_nonzero, replace=False)
            h_init[indices] = rng.rand(num_nonzero) + 0.1
        elif trial % 3 == 1:
            # Smooth random
            h_init = rng.rand(N)
            # Apply a simple smoothing filter
            kernel = np.array([1, 2, 1]) / 4.0
            h_init = np.convolve(h_init, kernel, mode='same')
        else:
            # Perturbed uniform
            h_init = np.ones(N) * (0.5 + 0.5 * rng.rand(N))
        
        # Normalize to avoid numerical issues (scale doesn't matter for ratio, but helps search bounds)
        if np.sum(h_init) > 0:
             h_init = h_init / np.sum(h_init) * N # Keep L1 ~ N
            
        h_opt, score_opt = _coordinate_descent(h_init, deadline, max_iters=20)
        
        if score_opt > best_score:
            best_score = score_opt
            best_heights = h_opt
    
    # Final polish on the best found
    if time.time() < deadline - 2.0:
        h_polish, score_polish = _coordinate_descent(best_heights, deadline, max_iters=50)
        if score_polish > best_score:
            best_score = score_polish
            best_heights = h_polish
    
    # Ensure nonnegative and return as list
    best_heights = np.maximum(best_heights, 0.0)
    
    if np.sum(best_heights) <= 1e-15:
        best_heights = np.ones(N)
    
    return [float(x) for x in best_heights]
