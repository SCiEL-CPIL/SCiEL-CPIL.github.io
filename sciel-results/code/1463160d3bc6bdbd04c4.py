"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum over shifts s in [0, 2] of the overlap integral.
    For a step function with N steps on [0, 2], the overlap for shift k*dw is:
    sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    We take the maximum over all k = 0, 1, ..., N-1.
    """
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _local_search_binary(h, max_iters, deadline):
    """
    Local search on the binary hypercube (h in {0,1}) with sum constraint.
    Swaps a 1 and a 0 if it reduces the max overlap.
    """
    h = h.copy()
    current_score = _max_overlap(h)
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    rng = np.random.default_rng(12345) # Local seed for variety
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Randomly pick a 1 and a 0 to swap
        idx_1 = rng.choice(ones)
        idx_0 = rng.choice(zeros)
        
        # Perform swap
        h[idx_1] = 0
        h[idx_0] = 1
        
        new_score = _max_overlap(h)
        
        if new_score < current_score:
            current_score = new_score
            # Update index sets
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
        else:
            # Revert
            h[idx_1] = 1
            h[idx_0] = 0
            
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_h = None
    best_score = np.inf
    
    # Use a few random restarts to escape local minima
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize binary sequence with N//2 ones
        h_init = np.zeros(N)
        ones_indices = np.random.choice(N, size=N//2, replace=False)
        h_init[ones_indices] = 1.0
        
        # Run local search
        h_res, score = _local_search_binary(h_init, max_iters=5000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_h = h_res
        
        # Optional: Try a simple anti-periodic binary initialization
        if restart % 2 == 1:
            h_anti = np.zeros(N)
            # First half 1s, second half 0s? No, sum must be N/2.
            # Anti-periodic: h[i] = 1 - h[i+N/2].
            # If we set first half to 1, second half to 0, sum is N/2.
            h_anti[:N//2] = 1.0
            h_anti[N//2:] = 0.0
            h_res2, score2 = _local_search_binary(h_anti, max_iters=5000, deadline=deadline)
            if score2 < best_score:
                best_score = score2
                best_h = h_res2

    if best_h is None:
        # Fallback: simple block
        best_h = np.zeros(N)
        best_h[:N//2] = 1.0
    
    # Ensure sum is exactly N/2 (it should be, but just in case)
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-12:
        diff = N/2.0 - current_sum
        # Adjust a few elements if needed (should not happen for binary)
        # For safety, just clip and adjust one element
        best_h = np.clip(best_h, 0.0, 1.0)
        current_sum = np.sum(best_h)
        if abs(current_sum - N/2.0) > 1e-12:
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h[idx] = np.clip(best_h[idx], 0.0, 1.0)
            
    return best_h
# EVOLVE-BLOCK-END
