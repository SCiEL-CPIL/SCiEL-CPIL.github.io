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

    # 1. Precompute primes and gcd helper
    # Max sum is 128 + 128 = 256
    max_val = 256
    
    # Sieve of Eratosthenes for quick prime check (primes are coprime to everything else)
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
            
    def is_pairwise_coprime(sums_set):
        """Check if all distinct sums are pairwise coprime."""
        # Optimization: If any sum is 1, it's coprime to all.
        # If any sum is even, no other sum can be even.
        # General check: O(k^2) where k is size of sums_set (max 9)
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

    # 2. Search Strategy
    # We want to maximize |A+B|. The maximum possible is 9.
    # We iterate through possible A and B.
    # To be efficient, we can iterate A, then for each A, iterate B.
    # However, 128^3 * 128^3 is too large (2^24 * 2^24 = 2^48).
    # We need a smarter search or pruning.
    
    # Let's try a heuristic search first, then fallback to exhaustive if time permits?
    # Actually, let's just do a structured exhaustive search with pruning.
    # 
    # Observation: If we pick A and B such that A+B contains many primes, it's likely valid.
    # Let's iterate over all combinations of A (3 from 128) and B (3 from 128).
    # C(128, 3) = 341,376.
    # Total pairs = 341,376^2 ~ 1.16 * 10^11. This is too slow for 85s in Python.
    
    # We need a much faster approach or a smarter search space.
    # Let's look for the maximum size 9 first. If found, return it.
    # If not, size 8, etc.
    
    # Alternative: Randomized search with local improvement?
    # Or, exploit structure.
    # If A = {a1, a2, a3} and B = {b1, b2, b3}.
    # Sums are ai + bj.
    
    # Let's try a different angle:
    # If we pick A and B to be subsets of primes, the sums are even (if both odd primes) or mixed.
    # Wait, if a, b are odd, a+b is even. Two even sums are not coprime.
    # So, if A and B consist of odd numbers, all 9 sums are even. 
    # If there are 2 distinct even sums, they share factor 2. 
    # So for size 9, all 9 sums must be distinct AND pairwise coprime.
    # If all sums are even, they can't be pairwise coprime unless there is only 1 distinct sum (impossible for distinct A,B usually) or one is 2 and others... no, 2 and 4 share 2.
    # Therefore, it is IMPOSSIBLE to have 9 pairwise coprime sums if all sums are even.
    # This implies at least one sum must be odd.
    # Sum is odd if one addend is even and the other is odd.
    # If A has an even number and B has an odd number, some sums are odd.
    
    # Let's try to find a solution with size 9 via random search + local optimization, 
    # as exhaustive is too slow.
    
    best_A = None
    best_B = None
    best_size = 0
    
    # Heuristic: Start with random sets, then try to swap elements to increase size and maintain validity.
    # But first, let's try a few specific structured candidates that are known to work well in similar problems.
    # Often, small primes work.
    
    # Let's run a bounded random search.
    import random
    random.seed(42) # For reproducibility in case we need to debug, though evaluator might not care
    
    max_iters = 500000 # Adjust based on time
    
    for i in range(max_iters):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        
        # Generate random A and B
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        
        sums = {a + b for a in A for b in B}
        
        # Quick check: if any even sum > 2 exists, and another even sum exists, fail.
        # Actually, just check pairwise coprime.
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    # Can't get better than 9
                    return [best_A, best_B]

    # If we didn't find 9, we might have found 8, 7, etc.
    # Let's also try a deterministic small search for high-quality candidates
    # e.g., A and B from small numbers
    small_range = range(1, 50)
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

    # Fallback to the seed solution if nothing better found
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END
