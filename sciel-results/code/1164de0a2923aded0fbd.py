"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Known good solution: all 9 sums are distinct primes
    # A=[1,7,13], B=[2,14,30] -> sums: 3,15,31,9,21,37,15,29,43
    # Wait, 15 appears twice. Let me verify a working solution.
    # A=[1,7,13], B=[2,10,16] -> 3,11,17,9,17,23,15,23,29 -> 17 and 23 repeat
    # Let's just use the search but with a fast path.
    
    # Quick check of a known valid configuration
    # We need a_i + b_j all prime and distinct for i,j in {0,1,2}
    # Try small values systematically first
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        i = 3
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True

    primes_set = set(x for x in range(2, 257) if is_prime(x))
    
    # Fast path: try small A and B values directly
    # This avoids the expensive precomputation if a small solution exists
    for a1 in range(1, 20):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 20):
            for a3 in range(a2 + 1, 20):
                A = [a1, a2, a3]
                for b1 in range(1, 20):
                    for b2 in range(b1 + 1, 20):
                        for b3 in range(b2 + 1, 20):
                            B = [b1, b2, b3]
                            sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                            if all(s in primes_set for s in sums) and len(set(sums)) == 9:
                                return [A, B]
    
    # If fast path failed, do the full search with precomputation
    # Precompute valid_b_pairs for all pairs
    valid_b_pairs = {}
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            valid_b = []
            for b in range(1, 129):
                if (a1 + b) in primes_set and (a2 + b) in primes_set:
                    valid_b.append(b)
            valid_b_pairs[(a1, a2)] = valid_b
    
    found_9 = False
    for a1 in range(1, 129):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline - 5:
                break
            vb_12 = valid_b_pairs.get((a1, a2), [])
            if len(vb_12) < 3:
                continue
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 5:
                    break
                valid_b_123 = [b for b in vb_12 if (a3 + b) in primes_set]
                if len(valid_b_123) < 3:
                    continue
                # Check if any 3 b's give 9 distinct sums
                b_cands = valid_b_123[:min(10, len(valid_b_123))]
                for b1, b2, b3 in combinations(b_cands, 3):
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    if len(set(sums)) == 9:
                        return [A, B]
    
    # Fallback: return a known valid (but maybe not max) solution
    # A=[1,7,13], B=[4,10,16] gives sums {5,11,17,11,17,23,17,23,29}
    # Distinct: {5,11,17,23,29} - all primes, pairwise coprime, count=5
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
