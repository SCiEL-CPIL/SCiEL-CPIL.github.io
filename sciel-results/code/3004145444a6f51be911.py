"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    def prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= prime_factors(arr[i] + arr[j])
        return len(primes)

    def is_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    def try_greedy_for_prime_set(prime_set, max_val=10000):
        """Greedy: build set of 12 numbers where all pairwise sums are smooth."""
        candidates = []
        for x in range(1, max_val + 1):
            if time_exceeded():
                return None
            valid = True
            for y in candidates:
                if not is_smooth(x + y, prime_set):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                return candidates[:12]
        return None

    # Strategy 1: Powers of 2 (structural)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    powers_of_2 = [2**i for i in range(12)]
    # Verify all <= 10000
    assert all(x <= 10000 for x in powers_of_2)
    assert len(set(powers_of_2)) == 12

    best_result = powers_of_2
    best_count = count_distinct_primes(powers_of_2)

    # Strategy 2: Try greedy for small prime sets
    prime_set_candidates = [
        {2},
        {2, 3},
        {3, 5},
        {2, 3, 5},
        {5, 7},
        {2, 5},
        {3, 7},
        {2, 3, 5, 7},
    ]

    for ps in prime_set_candidates:
        if time_exceeded():
            break
        result = try_greedy_for_prime_set(ps)
        if result is not None and len(result) == 12:
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                if best_count <= len(ps):
                    break

    # Strategy 3: Try arithmetic progressions with structured properties
    # If all numbers are multiples of M, sums are multiples of M.
    # Try: numbers = M * (small smooth numbers)
    # This doesn't directly help, but let's try specific structured sets.

    # Try: all odd numbers in a range, sums are even
    # Pick 12 odd numbers: 1,3,5,7,9,11,13,15,17,19,21,23
    # Sums are even, from 4 to 44. Factor those.
    odds = list(range(1, 24, 2))[:12]
    cnt = count_distinct_primes(odds)
    if cnt < best_count:
        best_count = cnt
        best_result = odds

    # Try: multiples of 6: 6,12,18,...,72
    # Sums are multiples of 6, from 18 to 138
    mult6 = [6*(i+1) for i in range(12)]
    cnt = count_distinct_primes(mult6)
    if cnt < best_count:
        best_count = cnt
        best_result = mult6

    # Try: numbers close together to limit sum range
    # 100,101,...,111 -> sums 201..222
    close = list(range(100, 112))
    cnt = count_distinct_primes(close)
    if cnt < best_count:
        best_count = cnt
        best_result = close

    # Try: geometric progression with ratio 3
    # 1,3,9,27,81,243,729,2187,6561 -> only 9, need 12
    # 1,3,9,...,3^11=177147 > 10000, so max 3^8=6561, that's 9 terms
    # Not enough. Try ratio 2 is already done.

    # Try: mix of small numbers
    small = list(range(1, 13))
    cnt = count_distinct_primes(small)
    if cnt < best_count:
        best_count = cnt
        best_result = small

    return best_result
# EVOLVE-BLOCK-END
