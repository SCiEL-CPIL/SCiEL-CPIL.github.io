"""Rewritten program to minimize the largest Sidon subset size for 12 integers in [0, 128]."""
import itertools
import time
import random

def _is_sidon(subset):
    """Check if a tuple has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given set."""
    # We check from largest possible size down to 1
    # For 12 elements, we check 12, 11, ..., 1
    # But we can stop early if we find one.
    # Optimization: We only care if it's < 4, or if it is 4.
    # Since 3 is likely impossible, we just need to confirm 4.
    
    # Check for size 12 down to 4
    for size in range(12, 3, -1):
        for combo in itertools.combinations(candidates, size):
            if _is_sidon(combo):
                return size
    # If no size >= 4 found, check size 3
    for combo in itertools.combinations(candidates, 3):
        if _is_sidon(combo):
            return 3
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate sets likely to have small max Sidon subsets
    # Dense sets (consecutive integers) maximize collisions.
    candidates = [
        list(range(0, 12)),
        list(range(1, 13)),
        list(range(2, 14)),
        list(range(0, 24, 2)), # Even numbers
        list(range(1, 25, 2)), # Odd numbers
        list(range(0, 36, 3)), # Multiples of 3
        list(range(0, 12, 1)), # Same as first
        list(range(100, 112)), # Shifted
    ]
    
    # Add some random dense sets
    random.seed(42)
    for _ in range(100):
        # Pick a random start and step 1
        start = random.randint(0, 116)
        candidates.append(list(range(start, start + 12)))
        
        # Pick a random dense cluster
        center = random.randint(0, 128)
        spread = random.randint(5, 15)
        vals = set()
        while len(vals) < 12:
            v = random.randint(max(0, center - spread), min(128, center + spread))
            vals.add(v)
        candidates.append(sorted(vals))

    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12 or max(cand) > 128 or min(cand) < 0:
            continue
        
        ms = _max_sidon_size(cand)
        if ms < best_size:
            best_size = ms
            best_set = cand
            if best_size <= 3:
                break # Can't do better than 3 (theoretically min is 4, but check just in case)

    return best_set
