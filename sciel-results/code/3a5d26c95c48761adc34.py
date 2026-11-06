from math import gcd
from itertools import combinations

def construct():
    # Precompute primes up to 256 (max sum 128+128)
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 17):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    best_count = 0
    best_result = None
    
    # Search for sets A, B such that all sums are prime.
    # This guarantees pairwise coprimality of distinct sums.
    # Using intersection method for efficiency.
    
    N = 100
    
    # Precompute valid b's for each a
    valid_b_for_a = {}
    for a in range(1, N + 1):
        valid_b_for_a[a] = [b for b in range(1, N + 1) if is_prime[a + b]]
        
    for A in combinations(range(1, N + 1), 3):
        # Find intersection of valid b's for all a in A
        valid_bs = set(valid_b_for_a[A[0]])
        for a in A[1:]:
            valid_bs.intersection_update(valid_b_for_a[a])
        
        if len(valid_bs) < 3:
            continue
            
        valid_bs_list = list(valid_bs)
        
        for B in combinations(valid_bs_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            count = len(sums)
            if count > best_count:
                best_count = count
                best_result = [list(A), list(B)]
                if best_count == 9:
                    return best_result
                    
    if best_result is None:
        return [[1, 2, 3], [1, 2, 3]]
        
    return best_result
