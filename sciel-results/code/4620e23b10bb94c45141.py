import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def get_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    small_primes = get_primes(100000)
    
    def is_prime(n):
        if n < 2: return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        if n % 5 == 0: return 5
        while True:
            c = 1
            x = 2
            y = 2
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factorize(n):
        if n == 1: return []
        if is_prime(n): return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def is_p_smooth(n, p):
        if n == 1: return True
        if n <= p: return True
        # Trial division by small primes
        for q in small_primes:
            if q > p: break
            if q * q > n: break
            while n % q == 0:
                n //= q
        if n == 1: return True
        if n <= p: return True
        # n > p and n > 1, check if n has a factor > p
        if is_prime(n):
            return n <= p
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

    def find_max_k(p, max_k=200):
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth(p2 + j, p):
                break
            k += 1
        return k

    best_k = 1
    best_p = 2

    # Generate primes in descending order from 10^9
    # Use segmented sieve in chunks
    def segmented_primes_desc(low, high):
        if high < 2:
            return []
        low = max(2, low)
        size = high - low + 1
        limit = int(math.isqrt(high)) + 1
        base_primes = get_primes(limit)
        is_prime_seg = bytearray(b'\x01') * size
        for p in base_primes:
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = 0
        return [low + i for i in range(size - 1, -1, -1) if is_prime_seg[i]]

    chunk_size = 50000
    high = 10**9
    low = high - chunk_size + 1
    
    while low >= 2:
        if time.time() > deadline:
            break
        primes_in_range = segmented_primes_desc(low, high)
        for p in primes_in_range:
            if time.time() > deadline:
                break
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
        high = low - 1
        low = high - chunk_size + 1
        if low < 2:
            low = 2
    
    return {'p': best_p, 'k': best_k}
