import itertools
import math
import time

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill

    max_val = 256
    
    # Sieve of Eratosthenes for quick prime check
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
            
    def is_pairwise_coprime(sums_set):
        """Check if all distinct sums are pairwise coprime."""
        sums_list = list(sums_set)
        n = len(sums_list)
        for i in range(n):
            si = sums_list[i]
            if si == 1:
                continue
            for j in range(i + 1, n):
                sj = sums_list[j]
                if math.gcd(si, sj) != 1:
                    return False
        return True

    best_A = None
    best_B = None
    best_size = 0

    # Strategy 1: Targeted search for prime-sum grids.
    # If all 9 sums are prime, they are distinct and pairwise coprime.
    # To get odd primes, we can use A=evens, B=odds (or vice versa).
    # We search small ranges first.
    
    even_candidates = [i for i in range(2, 31, 2)] # 2 to 30
    odd_candidates = [i for i in range(1, 31, 2)]   # 1 to 29
    
    for A in itertools.combinations(even_candidates, 3):
        if time.time() - t_start > TIME_LIMIT - 10:
            break
        for B in itertools.combinations(odd_candidates, 3):
            sums = {a + b for a in A for b in B}
            # Check if all sums are prime
            if all(is_prime[s] for s in sums):
                if len(sums) > best_size:
                    best_size = len(sums)
                    best_A = list(A)
                    best_B = list(B)
                    if best_size == 9:
                        return [best_A, best_B]
                        
    # Also try A=odds, B=evens
    for A in itertools.combinations(odd_candidates, 3):
        if time.time() - t_start > TIME_LIMIT - 10:
            break
        for B in itertools.combinations(even_candidates, 3):
            sums = {a + b for a in A for b in B}
            if all(is_prime[s] for s in sums):
                if len(sums) > best_size:
                    best_size = len(sums)
                    best_A = list(A)
                    best_B = list(B)
                    if best_size == 9:
                        return [best_A, best_B]

    # Strategy 2: General pairwise coprime check on small mixed sets.
    # This covers composite coprime sums.
    small_range = range(1, 51)
    for A in itertools.combinations(small_range, 3):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for B in itertools.combinations(small_range, 3):
            sums = {a + b for a in A for b in B}
            if is_pairwise_coprime(sums):
                if len(sums) > best_size:
                    best_size = len(sums)
                    best_A = list(A)
                    best_B = list(B)
                    if best_size == 9:
                        return [best_A, best_B]

    # Strategy 3: Random search with larger range if time permits.
    import random
    random.seed(42)
    max_iters = 300000
    for i in range(max_iters):
        if time.time() - t_start > TIME_LIMIT - 2:
            break
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        sums = {a + b for a in A for b in B}
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    return [best_A, best_B]

    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END
