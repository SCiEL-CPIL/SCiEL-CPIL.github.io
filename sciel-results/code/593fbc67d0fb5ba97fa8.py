"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fast_overlap_score(h):
    """
    Computes the maximum overlap of h with shifted 1-h over linear shifts.
    Uses np.correlate for speed.
    """
    one_minus_h = 1.0 - h
    # np.correlate(h, one_minus_h, 'valid') computes sum_{i=0}^{N-1-k} h[i] * one_minus_h[i+k]
    # This matches the definition in the original code.
    overlaps = np.correlate(h, one_minus_h, mode='valid')
    return np.max(overlaps)

def _local_search_binary(binary, time_limit=5.0):
    """
    Perform local search on a binary sequence to minimize max linear overlap.
    Uses swap-based moves to maintain sum = N/2.
    """
    current = binary.copy()
    current_score = _fast_overlap_score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        # Find a 1 and a 0 to swap
        ones = np.where(current == 1.0)[0]
        zeros = np.where(current == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        current[i] = 0.0
        current[j] = 1.0
        
        new_score = _fast_overlap_score(current)
        
        if new_score < best_score:
            best_score = new_score
            best = current.copy()
            current_score = new_score
        else:
            # Revert
            current[i] = 1.0
            current[j] = 0.0
            
    return best

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize with a random binary sequence
    # Ensure exactly N/2 ones
    binary = np.zeros(N)
    ones_indices = np.random.choice(N, N//2, replace=False)
    binary[ones_indices] = 1.0
    
    # Local search
    best_binary = _local_search_binary(binary, time_limit=10.0)
    best_score = _fast_overlap_score(best_binary)
    
    # Try a few more restarts if time permits
    while (time.time() - start_time) < deadline:
        binary2 = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        binary2[ones_indices] = 1.0
        
        # Allocate remaining time minus a small buffer for this restart
        time_left = deadline - (time.time() - start_time)
        if time_left < 1.0:
            break
            
        # Use a fraction of the remaining time for local search
        search_time = min(5.0, time_left * 0.8)
        cand = _local_search_binary(binary2, time_limit=search_time)
        score_cand = _fast_overlap_score(cand)
        
        if score_cand < best_score:
            best_binary = cand
            best_score = score_cand
            
    return best_binary

# EVOLVE-BLOCK-END
