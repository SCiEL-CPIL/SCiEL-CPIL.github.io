import time
import math
import random

def construct():
    deadline = time.time() + 110
    
    def is_prime_mr(n):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
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
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
            g = r = q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def factorize(n):
        if n <= 1:
            return []
        factors = []
        def _factor(m):
            if m <= 1:
                return
            if is_prime_mr(m):
                factors.append(m)
                return
            d = pollard_brent(m)
            _factor(d)
            _factor(m // d)
        _factor(n)
        return factors
    
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
        for d in small_primes:
            if d > p:
                return False
            while n % d == 0:
                n //= d
        if n == 1:
            return True
        if n <= p:
            return is_prime_mr(n)
        # n > p, need to check if all prime factors of n are <= p
        try:
            factors = factorize(n)
            for f in factors:
                if f > p:
                    return False
            return True
        except:
            return False
    
    # Precompute primes up to 10^6 for searching candidate p values
    sieve_limit = 10**6
    is_prime_arr = bytearray(b'\x01') * (sieve_limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = 0
    for i in range(2, int(math.isqrt(sieve_limit)) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
    
    primes = [i for i in range(2, sieve_limit + 1) if is_prime_arr[i]]
    
    best_k = 0
    best_p = 2
    
    # Search over primes
    for p in primes:
        if time.time() >= deadline - 5:
            break
        base = p * p
        k = 0
        while k < 200 and time.time() < deadline - 2:
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
