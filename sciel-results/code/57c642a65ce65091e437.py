"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import isqrt

def _is_smooth(n, primes):
    """Check if n is composed only of primes in the set primes."""
    if n <= 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
    return n == 1

def _all_pairs_smooth(values, primes):
    """Check if all pairwise sums of values are S-smooth."""
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if not _is_smooth(s, primes):
                return False
    return True

def construct():
    # We want to find 12 distinct positive integers <= 10000 such that
    # all pairwise sums have prime factors only from a small set S.
    # 
    # Strategy: Try small prime sets S and find 12 numbers whose pairwise sums are S-smooth.
    # 
    # Key insight: If we pick numbers that are all congruent to some residue r mod p for each p in S,
    # then a_i + a_j ≡ 2r mod p. For the sum to be divisible by p, we need 2r ≡ 0 mod p, i.e., r ≡ 0 mod p/2 (if p is odd) or r ≡ 0 mod 1 (trivial for p=2).
    # 
    # Actually, a simpler approach: pick numbers that are multiples of a large power of 2, so sums are also multiples of high powers of 2, leaving only small odd factors.
    # 
    # Even better: Use the fact that if all numbers are of the form k * 2^m for varying k, sums are 2^m * (k_i + k_j). If k_i + k_j is smooth over small primes, we're good.
    # 
    # Let's try: pick 12 numbers of the form 2^10 * k_i = 1024 * k_i where k_i are small distinct positive integers.
    # Then sum = 1024 * (k_i + k_j) = 2^10 * (k_i + k_j). The prime factors are {2} ∪ prime_factors(k_i + k_j).
    # If we pick k_i from {1, 2, ..., 12}, then k_i + k_j ranges from 3 to 23.
    # Prime factors of 3..23: 2,3,5,7,11,13,17,19,23. That's 9 primes plus 2 = 9 distinct primes total.
    # 
    # Can we do better? Let's try to minimize the set of primes in k_i + k_j.
    # 
    # Alternative: Use powers of 2 themselves: {1, 2, 4, 8, ..., 2^11}. Sums of two distinct powers of 2: 2^a + 2^b = 2^min(a,b) * (1 + 2^|a-b|).
    # 1 + 2^d is odd. For d=1: 3, d=2: 5, d=3: 9=3^2, d=4: 17, d=5: 33=3*11, d=6: 65=5*13, d=7: 129=3*43, d=8: 257 (prime), etc.
    # This introduces many primes.
    #
    # Let's try a different approach: search for a small prime set S and find 12 numbers.
    #
    # Try S = {2, 3, 5, 7}: 5-smooth numbers up to 10000.
    # 5-smooth numbers: 1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,...
    # We need 12 of these such that all pairwise sums are 5-smooth.
    #
    # Let's try a computational search with small prime sets.
    
    # Try prime sets in order of increasing size
    candidate_prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    for primes in candidate_prime_sets:
        # Generate all S-smooth numbers up to 10000
        smooth_numbers = set()
        def gen_smooth(remaining, p_idx, current):
            if p_idx == len(primes):
                if 1 <= current <= 10000:
                    smooth_numbers.add(current)
                return
            p = primes[p_idx]
            val = 1
            while current * val <= 10000:
                gen_smooth(remaining, p_idx + 1, current * val)
                val *= p
            # Also the case where we don't use this prime
            # Already handled by starting with val=1
        
        gen_smooth(10000, 0, 1)
        
        smooth_list = sorted(smooth_numbers)
        
        # We need to find 12 numbers from smooth_list such that all pairwise sums are S-smooth
        # Use backtracking
        n_needed = 12
        solution = []
        
        def backtrack(start_idx, current_list):
            if len(current_list) == n_needed:
                return True
            for i in range(start_idx, len(smooth_list)):
                candidate = smooth_list[i]
                # Check if candidate + existing elements are all S-smooth
                valid = True
                for existing in current_list:
                    s = candidate + existing
                    if not _is_smooth(s, primes):
                        valid = False
                        break
                if valid:
                    current_list.append(candidate)
                    if backtrack(i + 1, current_list):
                        return True
                    current_list.pop()
            return False
        
        if backtrack(0, solution):
            return solution
    
    # Fallback: use the simple approach
    return list(range(1, 13))
# EVOLVE-BLOCK-END
