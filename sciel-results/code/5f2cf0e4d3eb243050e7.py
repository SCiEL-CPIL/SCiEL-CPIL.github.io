"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)
    prime_set = set(primes)

    def is_prime_mr(n, a):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(a):
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def is_prime_large(n):
        if n < 10**6 + 1:
            return n in prime_set
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if not is_prime_mr(n, a):
                return False
        return True

    def segmented_phi_block(L, length):
        """
        Compute phi for all numbers in [L, L+length-1] using segmented sieve.
        Returns list of phi values.
        """
        n = length
        remaining = list(range(L, L + n))
        phi_val = list(range(L, L + n))
        
        for p in primes:
            if p * p > L + n - 1:
                break
            first = ((L + p - 1) // p) * p
            for j in range(first, L + n, p):
                idx = j - L
                if remaining[idx] % p == 0:
                    phi_val[idx] -= phi_val[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        for i in range(n):
            if remaining[i] > 1:
                phi_val[i] -= phi_val[i] // remaining[i]
        
        return phi_val

    def check_run_fast(s, max_k=200):
        """Check distinct phi run starting at s using segmented phi."""
        phi_vals = segmented_phi_block(s, max_k)
        seen = set()
        k = 0
        for v in phi_vals:
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def find_prime_near(target, search_range=10000):
        """Find a prime near target by searching downward."""
        # Ensure odd
        if target % 2 == 0:
            target -= 1
        for i in range(search_range // 2):
            candidate = target - 2 * i
            if candidate < 2:
                break
            if is_prime_large(candidate):
                return candidate
        return None

    best_k = 1
    best_s = 1

    # Strategy 1: Targeted prime-start search near 10^12
    # Primes have phi(p) = p-1, which is unique among small values
    # Starting at a prime gives a strong initial distinctness advantage
    target_max = 10**12 - 200
    
    # Try multiple prime starting points
    prime_search_starts = []
    # Search for primes in different regions
    for offset in [0, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]:
        region_start = target_max - offset
        if region_start < 10**11:
            region_start = 10**11
        prime = find_prime_near(region_start, search_range=5000)
        if prime and prime >= 1 and prime + 200 - 1 <= 10**12:
            prime_search_starts.append(prime)
    
    # Deduplicate and sort
    prime_search_starts = sorted(set(prime_search_starts), reverse=True)
    
    # Check each prime start
    for s in prime_search_starts:
        if time_left() < 30:
            break
        k = check_run_fast(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
    
    if best_k == 200:
        return {'start': best_s, 'length': best_k}
    
    # Strategy 2: Dense search near 10^12 (original approach)
    current = target_max
    if current % 2 == 0:
        current -= 1

    while current >= target_max - 200000 and time_left() > 30:
        k = check_run_fast(current, 200)
        if k > best_k:
            best_k = k
            best_s = current
            if best_k == 200:
                break
        current -= 2

    if best_k == 200:
        return {'start': best_s, 'length': best_k}

    # Strategy 3: Random samples in wide range
    if time_left() > 15:
        random.seed(42)
        for _ in range(5000):
            if time_left() < 5:
                break
            s = random.randint(10**11, 10**12 - 200)
            if s % 2 == 0:
                s += 1
            k = check_run_fast(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k == 200:
                    break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
