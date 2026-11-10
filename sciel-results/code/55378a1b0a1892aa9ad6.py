"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    SIZE = N_MAX + 1
    
    # Generate primes up to sqrt(N_MAX) for factorization
    sqrt_max = int(math.isqrt(N_MAX)) + 1
    sieve = [True] * (sqrt_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(sqrt_max)) + 1):
        if sieve[i]:
            for j in range(i*i, sqrt_max + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, sqrt_max + 1) if sieve[i]]
    
    # Generate powerful numbers (products of prime powers p^e, e>=2) up to N_MAX
    # with their prime support as a frozenset
    powerful_numbers = []  # list of (value, frozenset_of_primes)
    
    # For each prime, generate p^2, p^3, ... up to N_MAX
    prime_powers = []  # list of (prime, [p^2, p^3, ...])
    for p in primes:
        powers = []
        val = p * p
        while val <= N_MAX:
            powers.append(val)
            val *= p
            if val > N_MAX:
                break
        if powers:
            prime_powers.append((p, powers))
    
    # Build powerful numbers by combining prime powers from different primes
    # Start with single prime powers
    for p, powers in prime_powers:
        for pw in powers:
            if N_MIN <= pw <= N_MAX:
                powerful_numbers.append((pw, frozenset([p])))
    
    # Add products of two prime powers from different primes
    for i in range(len(prime_powers)):
        p1, powers1 = prime_powers[i]
        for j in range(i + 1, len(prime_powers)):
            p2, powers2 = prime_powers[j]
            for pw1 in powers1:
                for pw2 in powers2:
                    val = pw1 * pw2
                    if val <= N_MAX:
                        if N_MIN <= val <= N_MAX:
                            powerful_numbers.append((val, frozenset([p1, p2])))
                    else:
                        break
    
    # Add products of three prime powers from different primes
    for i in range(len(prime_powers)):
        p1, powers1 = prime_powers[i]
        for j in range(i + 1, len(prime_powers)):
            p2, powers2 = prime_powers[j]
            for k in range(j + 1, len(prime_powers)):
                p3, powers3 = prime_powers[k]
                for pw1 in powers1:
                    for pw2 in powers2:
                        base = pw1 * pw2
                        if base > N_MAX:
                            break
                        for pw3 in powers3:
                            val = base * pw3
                            if val <= N_MAX:
                                if N_MIN <= val <= N_MAX:
                                    powerful_numbers.append((val, frozenset([p1, p2, p3])))
                            else:
                                break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Remove duplicates and sort by value
    seen = set()
    unique_powerful = []
    for val, support in powerful_numbers:
        if val not in seen:
            seen.add(val)
            unique_powerful.append((val, support))
    
    unique_powerful.sort(key=lambda x: x[0])
    
    # For each powerful number A in range, check if A+1 has a powerful part B
    # such that A and B have disjoint prime supports
    # The powerful part of n is the product of p^e where e>=2 in n's factorization
    # We want to maximize pp(n) * pp(n+1)
    
    # Build a lookup: for each powerful number B, store its support
    # and for each n, we need to compute pp(n) and pp(n+1)
    
    # Actually, let's just do the full sieve approach but optimized
    # Build smallest prime factor sieve
    spf = list(range(SIZE))
    for i in range(2, int(math.isqrt(SIZE)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, SIZE, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [N_MIN, N_MAX]
    pp = [1] * SIZE
    for m in range(N_MIN, N_MAX + 1):
        x = m
        result = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                result *= (p ** e)
        pp[m] = result
    
    # Find best consecutive pair
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
