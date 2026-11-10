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
        # remaining[i] = remaining cofactor of (L+i) after removing small primes
        remaining = list(range(L, L + n))
        # phi_val[i] will accumulate the phi computation
        phi_val = list(range(L, L + n))  # start with phi(n) = n
        
        for p in primes:
            if p * p > L + n - 1:
                break
            # Find first multiple of p in [L, L+n-1]
            first = ((L + p - 1) // p) * p
            for j in range(first, L + n, p):
                idx = j - L
                if remaining[idx] % p == 0:
                    # p divides this number
                    phi_val[idx] -= phi_val[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # For each number, if remaining > 1, it's a prime factor
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

    best_k = 1
    best_s = 1

    # Strategy: Dense search near 10^12 where collisions are rare
    target_max = 10**12 - 200

    # Check consecutive odd numbers near the top
    # At 10^12, phi collisions in 200 consecutive numbers are extremely unlikely
    current = target_max
    if current % 2 == 0:
        current -= 1

    # Check blocks of odd numbers
    checked = 0
    while current >= target_max - 200000 and time_left() > 30:
        k = check_run_fast(current, 200)
        if k > best_k:
            best_k = k
            best_s = current
            if best_k == 200:
                break
        current -= 2
        checked += 1

    if best_k == 200:
        return {'start': best_s, 'length': best_k}

    # If not found near top, try random samples
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
