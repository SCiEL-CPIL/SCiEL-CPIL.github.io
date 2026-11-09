import itertools
import random
import time

def is_sidon(subset):
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def max_sidon_size(arr):
    n = len(arr)
    arr = tuple(sorted(arr))
    for k in range(n, 0, -1):
        for combo in itertools.combinations(range(n), k):
            subset = tuple(arr[i] for i in combo)
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    random.seed(42)
    
    while time.time() < deadline:
        candidate = random.sample(range(129), 12)
        size = max_sidon_size(candidate)
        if size < best_size:
            best_size = size
            best_set = candidate
            if best_size <= 4:
                break
    
    return best_set
