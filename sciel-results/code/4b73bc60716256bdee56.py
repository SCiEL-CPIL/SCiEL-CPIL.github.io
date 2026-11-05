import itertools
from math import gcd

def _pairwise_coprime(nums):
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    best_A = None
    best_B = None
    best_count = 0
    
    # Range [1, 128]
    # To ensure we find the best solution quickly, we prioritize
    # finding high-cardinality solutions.
    
    # 1. Try a small systematic search with pruning
    # 2. Try random search with more iterations if needed
    
    # Optimization: Check for early exit in coprime check
    def is_coprime_set(sums_set):
        lst = list(sums_set)
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    # Search in a moderate range first
    # Combinations of 3 from 1..30 is 4060. 4060^2 is 16M.
    # With pruning, this is fast.
    
    range_small = list(range(1, 31))
    
    for A in itertools.combinations(range_small, 3):
        A = list(A)
        for B in itertools.combinations(range_small, 3):
            B = list(B)
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            count = len(distinct_sums)
            if count > best_count:
                if is_coprime_set(distinct_sums):
                    best_count = count
                    best_A = A
                    best_B = B
                    if best_count == 9:
                        return [best_A, best_B]

    # If not found, try random in larger range
    import random
    random.seed(42)
    range_large = list(range(1, 129))
    
    for _ in range(100000):
        A = sorted(random.sample(range_large, 3))
        B = sorted(random.sample(range_large, 3))
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        if count > best_count:
            if is_coprime_set(distinct_sums):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]

    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
