"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_sidelobe(h):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation sidelobes.
    h: numpy array of length N containing 1s and -1s.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    # Aperiodic autocorrelation lags 1 to N-1 are in ac[1:N]
    # Due to symmetry, we only need to check positive lags
    return np.max(np.abs(ac[1:N]))

def _local_search(h, max_iters=5000, time_limit=10.0):
    """
    Perform local search to minimize the max sidelobe.
    """
    start_time = time.time()
    current_cost = _max_sidelobe(h)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the bit to flip that gives the biggest improvement
        best_improvement = 0
        best_idx = -1
        
        # We can't check all N bits every time if it's too slow, 
        # but N=400 is small. 400 FFTs of size 800 is fast.
        # However, doing this for every restart might be tight.
        # Let's try a random subset or just all if fast enough.
        
        # Optimization: Instead of full FFT for each flip, we can update cost incrementally?
        # For N=400, full FFT is ~10-50us. 400 flips = 2-20ms per iteration.
        # 5000 iters = 10-100s. Too slow for full scan every iter.
        
        # Strategy: Randomly sample a few bits to try, or use a heuristic.
        # Let's try sampling 10-20 random bits per iteration.
        
        indices = np.random.choice(N, size=min(20, N), replace=False)
        
        for idx in indices:
            h_new = h.copy()
            h_new[idx] *= -1
            new_cost = _max_sidelobe(h_new)
            
            if new_cost < current_cost - best_improvement:
                best_improvement = current_cost - new_cost
                best_idx = idx
                h_best = h_new
        
        if best_idx != -1:
            h = h_best
            current_cost = current_cost - best_improvement
        else:
            # No improvement found in sampled bits, might be stuck.
            # We could break, but random restarts handle this.
            # Or we can accept a neutral move to escape? 
            # Let's just break this local search phase.
            break
            
    return h, current_cost

def construct():
    start_time = time.time()
    time_budget = 110.0 # Leave margin for 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Random restarts
    while time.time() - start_time < time_budget:
        # Initialize with random signs
        h = np.random.choice([-1, 1], size=N)
        
        # Run local search with a small time slice
        # If we have 100s left, and each search takes ~1s, we can do ~100 restarts.
        # Let's allocate 5s per local search.
        h, score = _local_search(h, max_iters=2000, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_h = h
        
        # Check time
        if time.time() - start_time > time_budget - 5.0:
            break
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
