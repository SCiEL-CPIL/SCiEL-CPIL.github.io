"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a sorted tuple has all pair sums (i<=j) unique."""
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
    """Find the largest Sidon subset size of the given 12-element set."""
    # Try sizes from 12 down to 1
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            if _is_sidon(combo):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_set = None
    best_size = 13  # we want to minimize this
    
    # Pre-computed good candidate: arithmetic progression creates many sum collisions
    # Let's search among structured sets
    
    # Strategy 1: Try arithmetic progressions with various starts and steps
    # AP: {a, a+d, a+2d, ..., a+11d} all in [0,128]
    for d in range(1, 12):
        for a in range(0, 128 - 11*d + 1):
            if time.time() > deadline:
                break
            candidates = [a + i*d for i in range(12)]
            if max(candidates) > 128:
                continue
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 4:
                    return best_set
        if time.time() > deadline:
            break
    
    # Strategy 2: Try sets with many small numbers (dense in low range)
    # These create many sum collisions
    for start in range(0, 120):
        if time.time() > deadline:
            break
        for end in range(start + 11, min(start + 30, 129)):
            candidates = list(range(start, end + 1))
            if len(candidates) != 12:
                continue
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 4:
                    return best_set
    
    # Strategy 3: Random search with local improvement
    import random
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        candidates = random.sample(range(129), 12)
        ms = _max_sidon_size(candidates)
        if ms < best_size:
            best_size = ms
            best_set = candidates
            if best_size <= 4:
                return best_set
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
