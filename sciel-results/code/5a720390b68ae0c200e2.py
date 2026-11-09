"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset has unique pair sums (a+b for a<=b)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest size down
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 0

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 75.0  # Keep 15s buffer before 90s limit
    
    # Initialize with the simple AP which gives max Sidon = 4
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)  # should be 4
    
    # Try to find a set with max Sidon <= 3
    # Strategy: use sets with many 3-term APs or dense clustering
    random.seed(42)
    
    # Try structured sets first
    structured_candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # same as range(12)
        [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # mostly even
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],  # pairs with gaps
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # even numbers
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # dense
    ]
    
    for cand in structured_candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            s = max_sidon_size(cand)
            if s < best_score:
                best_score = s
                best_set = cand[:]
                if best_score <= 3:
                    break
    
    # Random search for max_sidon <= 3
    # Check all 4-subsets for Sidon property
    for attempt in range(20000):
        if time.time() - start_time > deadline:
            break
        
        # Generate random 12-element set in [0, 128]
        arr = sorted(random.sample(range(129), 12))
        
        # Quick check: does it have max Sidon <= 3?
        # Check all 4-subsets
        has_sidon4 = False
        for sub4 in combinations(arr, 4):
            if is_sidon(sub4):
                has_sidon4 = True
                break
        
        if not has_sidon4:
            # Verify it's really max Sidon <= 3
            actual_max = max_sidon_size(arr)
            if actual_max < best_score:
                best_score = actual_max
                best_set = arr[:]
                if best_score <= 3:
                    break
    
    # If we found max Sidon = 3, try to see if we can get even lower (unlikely)
    # Otherwise, return the best we found
    return best_set
# EVOLVE-BLOCK-END
