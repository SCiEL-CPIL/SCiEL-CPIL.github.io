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
    
    for A in itertools.combinations(even_as, 3):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        common_b = prime_b_for_a[A[0]] & prime_b_for_a[A[1]] & prime_b_for_a[A[2]]
        if len(common_b) < 3:
            continue
        b_list = sorted(common_b)
        for B in itertools.combinations(b_list, 3):
            # Check all 9 sums are distinct primes
            sums = set()
            valid = True
            for a in A:
                for b in B:
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid and len(sums) == 9:
                best_score = 9
                best_A = list(A)
                best_B = list(B)
                return [best_A, best_B]
    
    # Search Case 2: A odd, B even
    odd_as = [a for a in range(1, 129, 2)]
    
    for A in itertools.combinations(odd_as, 3):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        common_b = prime_b_for_a[A[0]] & prime_b_for_a[A[1]] & prime_b_for_a[A[2]]
        if len(common_b) < 3:
            continue
        b_list = sorted(common_b)
        for B in itertools.combinations(b_list, 3):
            # Check all 9 sums are distinct primes
            sums = set()
            valid = True
            for a in A:
                for b in B:
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid and len(sums) == 9:
                best_score = 9
                best_A = list(A)
                best_B = list(B)
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
