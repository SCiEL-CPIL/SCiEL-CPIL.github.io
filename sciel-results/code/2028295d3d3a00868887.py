"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    LIMIT = 10**18

    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def get_divisors(prime_list):
        """Generate all divisors of product of primes in prime_list (all exponent 1)."""
        divs = [1]
        for p in prime_list:
            divs = divs + [d * p for d in divs]
        divs.sort()
        return divs

    def score_from_divisors(divs):
        """Count adjacent coprime pairs in sorted divisors."""
        s = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                s += 1
        return s

    # Compute score for product of first 15 primes
    n_15 = 1
    for p in primes:
        n_15 *= p

    if n_15 <= LIMIT:
        divs_15 = get_divisors(primes)
        score_15 = score_from_divisors(divs_15)
        best_n = n_15
        best_score = score_15
    else:
        # Fallback to 14 primes
        n_14 = 1
        for p in primes[:14]:
            n_14 *= p
        divs_14 = get_divisors(primes[:14])
        score_14 = score_from_divisors(divs_14)
        best_n = n_14
        best_score = score_14

    # Check time - if we have time, try other combinations
    if time.time() - start_time < DEADLINE - 30:
        # Try dropping different primes from the 15-prime set
        for drop_idx in range(len(primes)):
            if time.time() - start_time > DEADLINE - 10:
                break
            test_primes = [p for i, p in enumerate(primes) if i != drop_idx]
            n_test = 1
            for p in test_primes:
                n_test *= p
            if n_test <= LIMIT and n_test >= 2:
                divs_test = get_divisors(test_primes)
                s = score_from_divisors(divs_test)
                if s > best_score:
                    best_score = s
                    best_n = n_test

    return best_n
# EVOLVE-BLOCK-END
