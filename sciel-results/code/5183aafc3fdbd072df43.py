import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_sidon(subset):
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def has_sidon_subset_of_size_k(S, k):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return True
        return False

    # 1. Exhaustive search for small K
    # K=16: 1820 sets
    # K=17: 2380 sets
    # K=18: 3060 sets
    # Total ~7000 sets.
    # Each check: 792 combos * ~20 ops = ~16000 ops.
    # Total ~110 million ops. Might be too slow in Python if not optimized.
    # But we can break early if found.
    
    for K in range(16, 20):
        if time.time() > deadline:
            break
        # Generate all subsets of size 12 from range(K)
        # This is equivalent to choosing 4 numbers to EXCLUDE from range(K)
        # Number of exclusions: C(K, 4)
        for exclude in combinations(range(K), 4):
            if time.time() > deadline:
                break
            exclude_set = set(exclude)
            S = [x for x in range(K) if x not in exclude_set]
            
            if not has_sidon_subset_of_size_k(S, 5):
                return S

    # 2. Random search for larger K
    for K in range(20, 50):
        if time.time() > deadline:
            break
        for _ in range(5000):
            if time.time() > deadline:
                break
            S = random.sample(range(K), 12)
            if not has_sidon_subset_of_size_k(S, 5):
                return S

    # Fallback
    return list(range(12))
