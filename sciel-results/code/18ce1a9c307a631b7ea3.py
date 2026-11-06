import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    # Precompute small primes
    SIEVE_LIMIT = 5_000_000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    primes_list = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in bases:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def check_p_smooth_batch(p, primes_list):
        """Check how many consecutive numbers p^2+1, p^2+2, ... are p-smooth using batch sieve."""
        p2 = p * p
        K = 200
        remaining = [p2 + j + 1 for j in range(K)]
        
        for q in primes_list:
            if q > p:
                break
            target = (-p2 - 1) % q
            if target >= K:
                continue
            for j in range(target, K, q):
                while remaining[j] % q == 0:
                    remaining[j] //= q
        
        k = 0
        for j in range(K):
            r = remaining[j]
            if r == 1:
                k += 1
            elif r <= p and is_prime_mr(r):
                k += 1
            else:
                break
        return k

    best_k = 1
    best_p = 2

    for p in primes_list:
        if time.time() - start_time > DEADLINE:
            break
        
        k = check_p_smooth_batch(p, primes_list)
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
