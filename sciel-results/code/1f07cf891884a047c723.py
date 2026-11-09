import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset is a Sidon set (all pair sums with repetition are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    """Find the largest Sidon subset size of arr."""
    n = len(arr)
    # Check from largest to smallest
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start = time.time()
    deadline = start + 70  # safety margin before 90s
    
    # First, try the simple AP which gives max Sidon = 4
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)  # should be 4
    
    # Try to find a set with max Sidon = 3
    # Strategy: use sets with many 3-term APs
    random.seed(42)
    
    # Try structured sets first
    # Idea: use a set where elements cluster in pairs close together
    # e.g., {0,1, 2,3, 4,5, 6,7, 8,9, 10,11} - same as range(12)
    
    # Try: all elements in a small range to force collisions
    # But range(12) already does that
    
    # Let's try random search for max_sidon = 3
    for attempt in range(5000):
        if time.time() > deadline:
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
            return arr
    
    # If we couldn't find max Sidon = 3, return the AP (max Sidon = 4)
    return best_set
