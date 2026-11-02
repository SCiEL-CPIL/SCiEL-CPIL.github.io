import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _local_search(signs, max_iters=20000, time_limit=120.0):
    """
    Local search: flip individual bits or small blocks to reduce the objective.
    Uses a greedy first-improvement strategy with random restarts within the search.
    """
    start_time = time.time()
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    # We will use a mix of single flips and small block flips
    # Pre-compute some useful structures
    # For efficiency, we can't easily pre-compute everything for 400 bits
    # but we can be smart about which moves to try.
    
    # Strategy: 
    # 1. Try all single flips (N=400 moves). If any improves, take the best one.
    # 2. If no single flip improves, try random pairs or blocks.
    # This is O(N^2) per iteration in the worst case if we compute score for each,
    # but score computation is O(N log N). 400 * 400 * log(400) is manageable?
    # 400 * 1000 ~ 400,000 operations per "full pass".
    # 20,000 iters * 400 moves * cost(fft) might be too slow.
    # Let's estimate cost of _compute_c3_bound.
    # N=400, FFT size 800.
    # np.fft.rfft(800) is very fast, maybe 10-20 microseconds?
    # So 400 moves * 20us = 8ms per full pass of single flips.
    # 1000 passes = 8 seconds. This is feasible within 120s.
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Phase 1: Single flips
        best_improvement = 0
        best_flip_idx = -1
        best_new_score = current_score
        
        # We can vectorize the evaluation of single flips?
        # Flipping bit j changes the autocorrelation.
        # R_new[k] = R_old[k] - 2*s_j*(s_{j+k} + s_{j-k}) ... roughly
        # It's easier to just loop and compute if N is small enough.
        # But 400 FFTs is ~8ms.
        
        for j in range(N):
            # Flip j
            current[j] = -current[j]
            new_score = _compute_c3_bound(current)
            current[j] = -current[j] # Flip back
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_improvement = current_score - new_score
                best_flip_idx = j
        
        if best_flip_idx != -1:
            # Apply the best flip
            current[best_flip_idx] = -current[best_flip_idx]
            current_score = best_new_score
            
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
        else:
            # No single flip improves. Try a random block flip of size 2 or 3.
            # This helps escape local minima.
            block_size = rng.choice([2, 3, 4])
            start = rng.randint(0, N - block_size + 1)
            # Flip block
            current[start:start+block_size] = -current[start:start+block_size]
            new_score = _compute_c3_bound(current)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                # Revert
                current[start:start+block_size] = -current[start:start+block_size]
                
            # If we've been stuck for a while (no improvement in last 100 iters of phase 1),
            # maybe do a larger random perturbation?
            # For now, just continue.
            
    return best, best_score

def _simulated_annealing(signs, start_temp=1.0, end_temp=0.01, n_iter=5000, block_sizes=None):
    """
    Simulated annealing with block-flip moves.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32]
    
    rng = np.random.RandomState(12345)
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        # Linear cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose a random block size with bias toward smaller blocks
        bs = block_sizes[rng.choice(len(block_sizes), p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])]
        if bs > N:
            bs = N
        
        start = rng.randint(0, N - bs + 1)
        # Flip block
        current[start:start+bs] = -current[start:start+bs]
        new_score = _compute_c3_bound(current)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
        else:
            # Revert
            current[start:start+bs] = -current[start:start+bs]
    
    return best, best_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize with random signs
    # All ones is a bad start (high autocorrelation)
    # Random is better
    best_pattern = rng.choice([1, -1], size=N)
    best_score = _compute_c3_bound(best_pattern)
    
    # Run local search from this start
    # Local search is powerful for this kind of problem.
    time_limit_ls = 60.0
    optimized, opt_score = _local_search(best_pattern, max_iters=50000, time_limit=time_limit_ls)
    
    if opt_score < best_score:
        best_score = opt_score
        best_pattern = optimized
    
    # If we have time left, run simulated annealing from the best found
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        sa_time_limit = DEADLINE - 5 - elapsed
        # Estimate iterations based on time
        # Each SA iteration is 1 FFT ~ 20us.
        # So 1 sec ~ 50,000 iters.
        n_iter_sa = int(sa_time_limit * 40000)
        n_iter_sa = max(n_iter_sa, 1000)
        
        optimized_sa, opt_score_sa = _simulated_annealing(
            best_pattern, 
            start_temp=1.0, 
            end_temp=0.01, 
            n_iter=n_iter_sa
        )
        
        if opt_score_sa < best_score:
            best_score = opt_score_sa
            best_pattern = optimized_sa
    
    result = np.sign(best_pattern).astype(int)
    
    # Constraint Check: heights must not sum to zero
    # If sum is 0, flip the first element to break the symmetry
    # This changes the score slightly but satisfies the constraint.
    if np.sum(result) == 0:
        result[0] = -result[0]
        
    return result
