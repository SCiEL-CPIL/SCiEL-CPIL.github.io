"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _linear_max_overlap(h):
    """
    Calculate the maximum linear overlap of h with shifted (1-h).
    h is a binary array (0 or 1) or real in [0,1].
    Overlap for shift k is sum_{i=k}^{N-1} h[i] * (1 - h[i-k]).
    """
    max_overlap = 0.0
    for k in range(N):
        if k == 0:
            # sum h[i]*(1-h[i])
            # For binary, this is 0.
            # For real, this is sum h[i] - h[i]^2
            overlap = np.sum(h * (1 - h))
        else:
            # h[k:] and h[:-k]
            h1 = h[k:]
            h2 = h[:-k]
            overlap = np.sum(h1 * (1 - h2))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _generate_random_binary(n, sum_target):
    """
    Generate a random binary sequence of length n with exactly sum_target ones.
    """
    h = np.zeros(n, dtype=np.float64)
    indices = np.random.choice(n, sum_target, replace=False)
    h[indices] = 1.0
    return h

def _local_search_binary(h, max_iter=2000, time_limit=100.0):
    """
    Local search on binary sequences.
    Move: pick two indices, one with 1 and one with 0, and swap them.
    """
    h = h.copy()
    current_score = _linear_max_overlap(h)
    best_h = h.copy()
    best_score = current_score
    
    start_time = time.time()
    
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Pick a random 1 and a random 0
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        # Swap
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = _linear_max_overlap(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            # Update lists
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
            
    return best_h, best_score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_h = None
    best_score = float('inf')
    
    # Baseline: constant 0.5
    h_const = np.full(N, 0.5)
    score_const = _linear_max_overlap(h_const)
    best_h = h_const
    best_score = score_const
    
    # Try random binary restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        h_rand = _generate_random_binary(N, N // 2)
        h_ls, score_ls = _local_search_binary(h_rand, max_iter=1000, time_limit=time_limit - (time.time() - start_time))
        
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls
    
    return best_h
# EVOLVE-BLOCK-END
