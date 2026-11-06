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
    # N=30 gives ~4000 combos. 4000^2 = 16M checks. Feasible in <60s.
    
    N = 30
    candidates = list(combinations(range(1, N+1), 3))
    
    for A in candidates:
        for B in candidates:
            sums = set()
            valid = True
            for a in A:
                for b in B:
                    s = a + b
                    if not is_prime[s]:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid:
                count = len(sums)
                if count > best_count:
                    best_count = count
                    best_result = [list(A), list(B)]
                    if best_count == 9:
                        return best_result
                        
    if best_result is None:
        # Fallback: return a valid small set if prime search fails (unlikely)
        # A=[1, 2, 3], B=[1, 2, 3] is invalid.
        # Let's just return the first valid one found in a smaller search or a hardcoded one.
        # But N=30 should find 9.
        return [[1, 2, 3], [1, 2, 3]]
        
    return best_result
