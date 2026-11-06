"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all 4096 subset sums distinct.
    Minimize the largest integer using greedy superincreasing construction.
    
    For distinct subset sums, a sufficient condition is that the sequence is superincreasing:
    a[i] > sum(a[0:i]) for all i. This gives the tightest possible lower bound on each element.
    """
    n = 12
    # Precompute subset masks as a binary matrix (4096, 12)
    masks = np.zeros((1 << n, n), dtype=np.int32)
    for i in range(n):
        masks[:, i] = (np.arange(1 << n) >> i) & 1

    # Greedy superincreasing: a[0]=1, a[i] = sum(a[0:i]) + 1
    # This gives: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    seq = [1]
    for i in range(11):
        seq.append(sum(seq) + 1)
    
    best_seq = seq[:]
    best_max = max(seq)
    
    # Try to improve via local search
    deadline = time.time() + 85  # 85 second deadline (5 sec safety margin)
    
    def all_subset_sums_distinct(arr):
        """Check if all 2^n subset sums are distinct using vectorized numpy."""
        # arr is a list, convert to numpy array for matmul
        # We can avoid conversion if we keep seq as numpy, but list is easier for modification
        arr_np = np.array(arr, dtype=np.int32)
        sums = masks @ arr_np
        # Check if all sums are unique
        # Sorting and checking diffs is fast, or np.unique
        return len(np.unique(sums)) == len(sums)
    
    # Local search: try to decrease elements while maintaining validity
    # We will focus on decreasing the largest elements first to reduce max quickly
    # But random search is also good. Let's stick to random but optimize the check.
    
    for iteration in range(200000): # Increased iterations due to speedup
        if time.time() > deadline:
            break
        
        # Try to decrease a random element
        idx = random.randint(0, 11)
        if idx == 0:
            continue  # Can't decrease a[0] below 1
        
        current_val = seq[idx]
        if current_val <= 1:
            continue
        
        # Try to decrease by 1
        new_val = current_val - 1
        if new_val <= 0:
            continue
        
        # Check for duplicates in the sequence itself (distinct integers required)
        if new_val in seq:
            continue
        
        old_val = seq[idx]
        seq[idx] = new_val
        
        if all_subset_sums_distinct(seq):
            new_max = max(seq)
            if new_max < best_max:
                best_max = new_max
                best_seq = seq[:]
        else:
            seq[idx] = old_val
    
    return best_seq
# EVOLVE-BLOCK-END
