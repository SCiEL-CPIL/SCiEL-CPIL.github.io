"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute GCD for numbers up to 256
    MAX_SUM = 256
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) > 1:
                    return False
        return True
    
    # Strategy: Search for A, B such that all 9 sums are distinct and pairwise coprime.
    # Key insight: if all sums are primes, they're automatically pairwise coprime.
    # We want a 3x3 addition table where all entries are distinct primes.
    
    # Precompute primes up to 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, MAX_SUM + 1) if is_prime[p]]
    
    best_result = [[1, 7, 13], [4, 10, 16]]
    best_score = 5  # cardinality of seed
    
    # Try to find configurations where all 9 sums are distinct primes
    # A = [a1, a2, a3], B = [b1, b2, b3]
    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # Heuristic: use small values for A and B to keep sums in prime range
    # Try systematic search with time limit
    
    # First, try a focused search for all-prime sumsets
    # A elements should be small, B elements should be small
    
    found_optimal = False
    candidates_a = []
    candidates_b = []
    
    # Generate candidate values: we want sums to be primes
    # If A = [a1, a2, a3] and B = [b1, b2, b3], all 9 sums must be prime
    # This is a strong constraint. Let's try small values.
    
    # Try all combinations of small values first
    small_range = list(range(1, 50))
    
    # Use itertools but with early termination
    count = 0
    for a1 in small_range:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in small_range:
            if a2 <= a1:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in small_range:
                if a3 <= a2:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                A = [a1, a2, a3]
                
                for b1 in small_range:
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                    for b2 in small_range:
                        if b2 <= b1:
                            continue
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b3 in small_range:
                            if b3 <= b2:
                                continue
                            if time.time() - t_start > TIME_LIMIT - 5:
                                break
                            B = [b1, b2, b3]
                            
                            # Compute all 9 sums
                            sums = [a + b for a in A for b in B]
                            
                            # Check if all are distinct
                            if len(set(sums)) < 9:
                                continue
                            
                            # Check if all are prime (stronger than pairwise coprime)
                            if all(is_prime[s] for s in sums):
                                best_result = [A, B]
                                best_score = 9
                                found_optimal = True
                                break
                        if found_optimal:
                            break
                    if found_optimal:
                        break
                if found_optimal:
                    break
            if found_optimal:
                break
        if found_optimal:
            break
    
    if found_optimal:
        return best_result
    
    # If no all-prime solution found, try to maximize cardinality with pairwise coprime
    # Try larger ranges and check pairwise coprime instead of all-prime
    count2 = 0
    for a1 in range(1, 30):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in range(a1 + 1, 30):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in range(a2 + 1, 30):
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                A = [a1, a2, a3]
                
                for b1 in range(1, 30):
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                    for b2 in range(b1 + 1, 30):
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b3 in range(b2 + 1, 30):
                            if time.time() - t_start > TIME_LIMIT - 5:
                                break
                            B = [b1, b2, b3]
                            
                            sums = [a + b for a in A for b in B]
                            distinct_sums = list(set(sums))
                            card = len(distinct_sums)
                            
                            if card > best_score and is_pairwise_coprime(distinct_sums):
                                best_score = card
                                best_result = [A, B]
                                if card == 9:
                                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
