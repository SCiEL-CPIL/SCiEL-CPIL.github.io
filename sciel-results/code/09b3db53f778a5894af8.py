"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
import random

N = 100
DEADLINE = 145.0  # Safety margin
_START = time.time()

def remaining_time():
    return DEADLINE - (time.time() - _START)

def compute_value(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def construct():
    """
    Strategy: Discrete Binary Simulated Annealing / Hill Climbing.
    
    We restrict the search space to binary vectors h in {0, 1}^N with exactly 
    N/2 ones. This is a valid subset of the feasible region [0,1]^N with sum N/2.
    Often, the optimal solution for minimizing max overlap is binary.
    
    We use a fast incremental update for the correlation vector to allow 
    thousands of swaps.
    
    1. Initialize h as a random binary vector with 50 ones.
    2. Compute initial correlation c = correlate(h, 1-h, 'full').
    3. Perform swaps: pick two indices i, j. Swap h[i] and h[j].
       Update c efficiently.
       If max(c) decreases, accept.
       If max(c) increases, accept with probability exp(-delta/T).
    4. Cool down T.
    5. Return the best h found.
    """
    
    # Initial best: Uniform 0.5 (score 1.0)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Binary initialization
    # Create a random binary vector with exactly N//2 ones
    h = np.zeros(N)
    ones_indices = random.sample(range(N), N // 2)
    h[ones_indices] = 1.0
    
    # Compute initial correlation
    # c_k = sum_{i=0}^{N-1-k} h[i] * (1-h[i+k])
    # Note: np.correlate(h, 1-h, 'full') returns c for lags -(N-1) to N-1.
    # Let's stick to the exact definition in the evaluator.
    
    def get_corr(h):
        return np.correlate(h, 1.0 - h, mode="full")
    
    c = get_corr(h)
    current_val = np.max(c) / N * 2.0
    
    if current_val < best_val:
        best_val = current_val
        best_h = h.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9999
    min_T = 1e-6
    
    # We need to track the current max index to update it efficiently?
    # Actually, just recomputing max(c) is O(N). 
    # Updating c for a swap is O(N).
    # So each step is O(N). With N=100, this is very fast.
    # We can do ~100,000 steps easily.
    
    # Helper to update correlation after swapping h[i] and h[j]
    # Let's derive the delta.
    # c_k = sum_{m} h[m] (1 - h[m+k])
    # If we swap h[i] and h[j], let h_new[i] = h_old[j], h_new[j] = h_old[i].
    # Delta c_k = sum_m (h_new[m] - h_old[m]) (1 - h_new[m+k]) + h_old[m] (h_old[m+k] - h_new[m+k])
    # This is complex. Since N=100 is small, we can just recompute c from scratch 
    # or use a slightly smarter update.
    # Recomputing c is O(N^2) if done naively via loops, but np.correlate is O(N^2) 
    # or O(N log N) with FFT? For N=100, it's negligible.
    # Let's just recompute c using np.correlate for simplicity and robustness, 
    # unless we find it's too slow.
    
    # Actually, let's try to be a bit smarter.
    # c_k = sum_{m=0}^{N-1-k} h_m (1-h_{m+k})
    # Let's just use np.correlate. It's C-optimized.
    
    steps = 0
    last_best_update = time.time()
    
    while remaining_time() > 5.0:
        # Pick random i, j
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
            
        # Check if swap changes the vector (only if h[i] != h[j])
        if h[i] == h[j]:
            continue
            
        # Perform swap
        h[i], h[j] = h[j], h[i]
        
        # Compute new value
        c_new = get_corr(h)
        new_max = np.max(c_new)
        new_val = new_max / N * 2.0
        
        delta_val = new_val - current_val
        
        # Acceptance criterion
        if delta_val < 0 or (T > min_T and random.random() < np.exp(-delta_val / T)):
            # Accept
            current_val = new_val
            c = c_new
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
                last_best_update = time.time()
                
            # Cool down
            T *= cooling
        else:
            # Reject, swap back
            h[i], h[j] = h[j], h[i]
            
        steps += 1
        
        # If we've been stuck for a long time, maybe restart with a different seed?
        # Or just keep going.
        
    # Final check
    # Ensure best_h is valid
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
        
    # The problem allows h in [0,1]. Our solution is binary, which is valid.
    # Sum is exactly N/2.
    
    return best_h

# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
