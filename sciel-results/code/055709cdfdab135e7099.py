import time
import random
from itertools import combinations

def is_sidon_4(subset):
    # subset is a tuple of 4 distinct integers
    # Check if all unordered pair sums (including diagonals) are unique
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(a_set):
    # For 12 elements, max possible is 12.
    # We check sizes from 12 down to 1? No, too slow.
    # We only care if it's <= 3 or > 3.
    # If any 4-subset is Sidon, max size >= 4.
    # If no 4-subset is Sidon, max size <= 3.
    # We assume max size is at least 3 (most 3-sets are Sidon).
    # So we just return 3 if no 4-set is Sidon, else 4+ (we don't need exact if >3).
    for comb in combinations(a_set, 4):
        if is_sidon_4(comb):
            return 4 # Actually could be higher, but we just want to know it's not 3
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    best_set = list(range(12))
    best_score = 4 # Baseline
    
    # Try to find a set with max Sidon size 3
    # Search in small ranges to maximize collisions
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        # Random range size, biased towards small
        r = random.randint(12, 40)
        # Generate 12 distinct ints in [0, r]
        if r < 11:
            continue
        s = random.sample(range(r + 1), 12)
        
        # Check if any 4-subset is Sidon
        has_sidon_4 = False
        for comb in combinations(s, 4):
            if is_sidon_4(comb):
                has_sidon_4 = True
                break
        
        if not has_sidon_4:
            # Found a set with max Sidon size 3 (likely)
            return s
            
    return best_set
