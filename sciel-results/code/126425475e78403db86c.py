import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Normalize to L1=1
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def local_search(h, max_iters=2000, time_limit=10.0):
    """
    Perform local search by transferring mass between pairs of indices.
    h is a probability vector (sum=1).
    """
    h = h.copy()
    current_score = compute_ratio(h)
    start_time = time.time()
    
    # Precompute indices for faster convolution? 
    # np.convolve is fast enough for N=50.
    
    for _ in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        
        # Pick two random indices
        i, j = np.random.choice(N, 2, replace=False)
        
        # Determine transfer amount
        # Try to move mass from j to i if h_j is large, or i to j if h_i is large?
        # Just try a small random perturbation in both directions.
        # Let's try moving a fraction of h_j to i.
        eps = h[j] * 0.1 * np.random.rand()
        
        h_new = h.copy()
        h_new[i] += eps
        h_new[j] -= eps
        
        if h_new[j] < 0:
            continue
            
        new_score = compute_ratio(h_new)
        
        if new_score > current_score:
            h = h_new
            current_score = new_score
        else:
            # Try reverse direction (move from i to j)
            eps2 = h[i] * 0.1 * np.random.rand()
            h_new2 = h.copy()
            h_new2[j] += eps2
            h_new2[i] -= eps2
            
            if h_new2[i] < 0:
                continue
                
            new_score2 = compute_ratio(h_new2)
            if new_score2 > current_score:
                h = h_new2
                current_score = new_score2
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial best candidate
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Candidate 1: Uniform
    h_uni = np.ones(N) / N
    score_uni = compute_ratio(h_uni)
    if score_uni > best_score:
        best_score = score_uni
        best_h = h_uni.copy()
        
    # Candidate 2: Delta
    # Already initialized
    
    # Run multiple restarts of local search
    # We can afford many restarts if each is fast.
    # N=50, conv is O(N^2) ~ 2500 ops. Very fast.
    # Let's run until deadline.
    
    np.random.seed(42) # Reproducibility for debugging, though random restarts are key
    
    while time.time() < deadline - 5:
        # Generate a random initial candidate
        # Mix of random and structured
        r = np.random.rand()
        if r < 0.3:
            # Sparse random
            h_init = np.zeros(N)
            num_nonzero = np.random.randint(1, 10)
            indices = np.random.choice(N, num_nonzero, replace=False)
            h_init[indices] = 1.0
            h_init = h_init / np.sum(h_init)
        elif r < 0.6:
            # Gaussian-like
            center = np.random.randint(0, N)
            width = np.random.uniform(1, 10)
            x = np.arange(N)
            h_init = np.exp(-((x - center)**2) / (2 * width**2))
            h_init = h_init / np.sum(h_init)
        else:
            # Uniform random
            h_init = np.random.rand(N)
            h_init = h_init / np.sum(h_init)
            
        # Run local search
        h_opt, score_opt = local_search(h_init, max_iters=5000, time_limit=5.0)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
        # Also try a "smooth" version of the best found so far?
        # Maybe blur the best_h slightly and re-optimize?
        if best_score > 1.01:
            h_smooth = np.convolve(best_h, np.ones(3)/3, mode='same')
            h_smooth = h_smooth / np.sum(h_smooth)
            h_opt2, score_opt2 = local_search(h_smooth, max_iters=2000, time_limit=3.0)
            if score_opt2 > best_score:
                best_score = score_opt2
                best_h = h_opt2.copy()

    return list(best_h)
