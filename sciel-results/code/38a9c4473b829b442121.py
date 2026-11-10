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

    # Miller-Rabin for large primes
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
        # Deterministic Miller-Rabin for n < 3.2 * 10^18
        # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if not is_prime_mr(n, a):
                return False
        return True

    phi_cache = {}

    def phi_fast(n):
        if n in phi_cache:
            return phi_cache[n]
        if n == 1:
            phi_cache[n] = 1
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        phi_cache[n] = result
        return result

    def check_run(s, max_k=200):
        """Check if phi(s), ..., phi(s+max_k-1) are distinct. Returns length of distinct prefix."""
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Strategy: Search for prime anchors near 10^12
    # Start from the top and go down
    # We want s + k - 1 <= 10^12, so s <= 10^12 - 199
    target_max = 10**12 - 200
    
    # Check a dense range near the top
    # We will check candidates in descending order
    # Step 1: Find primes in [target_max - 100000, target_max]
    # Using is_prime_large is fast enough for this range
    
    search_range = 200000
    start_search = target_max
    end_search = target_max - search_range
    
    # Iterate through candidates
    # To save time, we can skip even numbers > 2
    current = start_search
    if current % 2 == 0:
        current -= 1
        
    found_prime = False
    while current >= end_search and time_left() > 20:
        if is_prime_large(current):
            k = check_run(current, 200)
            if k > best_k:
                best_k = k
                best_s = current
                if best_k == 200:
                    found_prime = True
                    break
        current -= 2
        
    if found_prime:
        return {'start': best_s, 'length': best_k}
        
    # If we didn't find a good run in the top range, try random samples in a wider range
    # or just accept the best found so far if time is low
    if time_left() > 10:
        # Try some random starts in [10^11, 10^12]
        # Focus on numbers that might be primes or near primes
        random.seed(42)
        for _ in range(500):
            if time_left() < 5:
                break
            # Pick a random odd number
            s = random.randint(10**11, 10**12 - 200)
            if s % 2 == 0:
                s += 1
            
            # Check if it's a prime, if so it's a good candidate
            # Or just check the run directly?
            # Checking run directly is expensive if it's long, but we break early on collision.
            # However, if it's not a prime, the run might be short.
            # Let's prioritize primes.
            if is_prime_large(s):
                k = check_run(s, 200)
                if k > best_k:
                    best_k = k
                    best_s = s
                    if best_k == 200:
                        break
            else:
                # Maybe check s-1 or s+1 if they are primes?
                # Or just skip non-primes to save time?
                # Non-primes can also start long runs, but primes are better "anchors"
                pass

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
