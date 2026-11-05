import time
import random
import itertools

def is_sidon_fast(subset):
    """Check if a subset has all unique pair sums including diagonals."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size_fast(A):
    """Compute the maximum size of a Sidon subset in A."""
    n = len(A)
    for k in range(n, 3, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon_fast(combo):
                return k
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_candidate = list(range(12))
    best_size = max_sidon_size_fast(best_candidate)
    
    # Try contiguous blocks
    for start in range(0, 117):
        if time.time() > deadline:
            break
        candidate = list(range(start, start + 12))
        size = max_sidon_size_fast(candidate)
        if size < best_size:
            best_size = size
            best_candidate = candidate
    
    # Try arithmetic progressions
    for d in range(1, 11):
        if time.time() > deadline:
            break
        max_start = 128 - 11 * d
        for start in range(0, max_start + 1):
            if time.time() > deadline:
                break
            candidate = [start + i * d for i in range(12)]
            size = max_sidon_size_fast(candidate)
            if size < best_size:
                best_size = size
                best_candidate = candidate
    
    # Try random sets
    random.seed(12345)
    attempts = 0
    while time.time() < deadline and attempts < 20000:
        attempts += 1
        candidate = sorted(random.sample(range(129), 12))
        size = max_sidon_size_fast(candidate)
        if size < best_size:
            best_size = size
            best_candidate = candidate
    
    return best_candidate
