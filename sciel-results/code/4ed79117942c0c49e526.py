"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all 4096 subset sums distinct.
    Minimize the largest integer.
    
    Strategy:
    1. Start with the tightest superincreasing sequence (powers of 2): 1, 2, 4, ..., 2048.
    2. Use vectorized numpy subset-sum checks for fast validity testing.
    3. Perform local search to reduce the largest element by:
       a. Trying to decrease individual elements (especially the largest).
       b. Trying random replacements to escape local minima.
    4. Return the best valid sequence found.
    """
    n = 12
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute subset masks as a binary matrix (4096, 12)
    # masks[i, j] = 1 if element j is in subset i
    masks = np.zeros((1 << n, n), dtype=np.int32)
    for i in range(n):
        masks[:, i] = (np.arange(1 << n) >> i) & 1
    
    def all_subset_sums_distinct(arr):
        """Check if all 2^n subset sums are distinct using vectorized numpy."""
        arr_np = np.array(arr, dtype=np.int64)
        sums = masks @ arr_np
        # Check if all sums are unique
        return len(np.unique(sums)) == len(sums)
    
    # Start with powers of 2 (superincreasing, known valid)
    seq = [1 << i for i in range(n)]
    best_seq = seq[:]
    best_max = max(seq)  # 2048
    
    # Local search: try to decrease elements while maintaining validity
    # Focus on decreasing the largest elements first
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Determine which index to modify
        # Bias towards the largest element to reduce max quickly
        if random.random() < 0.5:
            # Pick the index of the current maximum
            idx = max(range(n), key=lambda i: seq[i])
        else:
            idx = random.randint(0, n - 1)
        
        if idx == 0:
            continue  # Can't decrease a[0] below 1
        
        current_val = seq[idx]
        if current_val <= 1:
            continue
        
        # Try to decrease by a random amount (1 to min(5, current_val-1))
        max_decrease = min(5, current_val - 1)
        decrease = random.randint(1, max_decrease)
        new_val = current_val - decrease
        
        if new_val <= 0:
            continue
        
        # Check for duplicates in the sequence itself
        if new_val in seq:
            continue
        
        old_val = seq[idx]
        seq[idx] = new_val
        
        if all_subset_sums_distinct(seq):
            new_max = max(seq)
            if new_max < best_max:
                best_max = new_max
                best_seq = seq[:]
            # Accept the move even if max didn't decrease, to explore
        else:
            seq[idx] = old_val
    
    return best_seq
# EVOLVE-BLOCK-END
