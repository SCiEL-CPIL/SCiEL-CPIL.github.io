"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill

    # Precompute sieve for primality up to 256 (max sum = 128+128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def count_distinct_coprime(A, B):
        """Count distinct values in A+B that are pairwise coprime. Return 0 if not all are pairwise coprime."""
        sums = set(a + b for a in A for b in B)
        if is_pairwise_coprime(sums):
            return len(sums)
        return 0

    # Best known result from seed: 5 distinct coprime sums
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Strategy: Try to find 9 distinct primes as sums (maximum possible for 3x3)
    # If we can find A and B such that all 9 sums are distinct primes, that's optimal.
    
    # Key insight: If all 9 sums are primes, they're automatically pairwise coprime.
    # Parity constraint: To have all sums be odd primes (>2), A and B must have opposite parity.
    # Case 1: A even, B odd.
    # Case 2: A odd, B even.
    
    # Precompute for each a in [1,128], the set of b in [1,128] where a+b is prime.
    prime_b_for_a = {}
    for a in range(1, 129):
        prime_b_for_a[a] = set()
        for b in range(1, 129):
            if is_prime[a + b]:
                prime_b_for_a[a].add(b)
    
    # Search Case 1: A even, B odd
    even_as = [a for a in range(2, 129, 2)]
    odd_bs = [b for b in range(1, 129, 2)]
    
    # To make search efficient, we iterate over triples of A.
    # We limit the search to a subset if it takes too long, but 128 is small enough for optimized loops.
    
    for a1_idx, a1 in enumerate(even_as):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2_idx, a2 in enumerate(even_as):
            if a2_idx <= a1_idx:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3_idx, a3 in enumerate(even_as):
                if a3_idx <= a2_idx:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                
                # Find b values that make all sums with a1, a2, a3 prime
                # Since we restricted B to odd in the set definition implicitly by checking prime_b_for_a,
                # but prime_b_for_a contains ALL b. We should intersect with odd_bs?
                # Actually, if a is even, a+b is prime (odd) implies b is odd.
                # So prime_b_for_a[a] already only contains odd b's (except if a+b=2, but a>=2,b>=1 -> sum>=3).
                # So the intersection will naturally contain only odd b's.
                
                common_b = prime_b_for_a[a1] & prime_b_for_a[a2] & prime_b_for_a[a3]
                
                if len(common_b) < 3:
                    continue
                
                # Now try all triples of b from common_b
                b_list = sorted(common_b)
                
                # Optimization: If common_b is large, we might want to sample or limit.
                # But let's try all first.
                for b1_idx, b1 in enumerate(b_list):
                    for b2_idx, b2 in enumerate(b_list):
                        if b2_idx <= b1_idx:
                            continue
                        for b3_idx, b3 in enumerate(b_list):
                            if b3_idx <= b2_idx:
                                continue
                            
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            # Check all 9 sums are distinct primes
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums or not is_prime[s]:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                best_score = 9
                                best_A = A
                                best_B = B
                                return [best_A, best_B]
    
    # Search Case 2: A odd, B even
    odd_as = [a for a in range(1, 129, 2)]
    even_bs = [b for b in range(2, 129, 2)]
    
    for a1_idx, a1 in enumerate(odd_as):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2_idx, a2 in enumerate(odd_as):
            if a2_idx <= a1_idx:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3_idx, a3 in enumerate(odd_as):
                if a3_idx <= a2_idx:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                
                common_b = prime_b_for_a[a1] & prime_b_for_a[a2] & prime_b_for_a[a3]
                
                if len(common_b) < 3:
                    continue
                
                b_list = sorted(common_b)
                
                for b1_idx, b1 in enumerate(b_list):
                    for b2_idx, b2 in enumerate(b_list):
                        if b2_idx <= b1_idx:
                            continue
                        for b3_idx, b3 in enumerate(b_list):
                            if b3_idx <= b2_idx:
                                continue
                            
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums or not is_prime[s]:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                best_score = 9
                                best_A = A
                                best_B = B
                                return [best_A, best_B]
    
    # Fallback: If no 9-prime solution found, search for best pairwise coprime sum
    # Use a smaller range for brute force to ensure completion
    range2 = range(1, 40)
    for A in itertools.combinations(range2, 3):
        if time.time() - t_start > TIME_LIMIT - 3:
            break
        for B in itertools.combinations(range2, 3):
            score = count_distinct_coprime(list(A), list(B))
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                if best_score == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
