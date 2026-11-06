import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115
    
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
        g, r, q = 1, 1, 1
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
        return g

    def has_large_factor(n, p):
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for q in small_primes:
            if n % q == 0:
                if q > p:
                    return True
                while n % q == 0:
                    n //= q
                if n == 1:
                    return False
        
        if n == 1:
            return False
            
        if is_prime(n):
            return n > p
            
        # Find a factor
        f = pollard_brent(n)
        if f > p:
            return True
        n //= f
        # Check cofactor
        # The cofactor might be composite or prime
        # Recursively check? Or just check if cofactor has large factor.
        # Since f <= p, we need to check n (which is now n/f).
        # If n/f > p and is_prime(n/f), then True.
        # If n/f is composite, it might have factor > p.
        # So we call has_large_factor again?
        # But n is getting smaller.
        return has_large_factor(n, p)

    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    best_p = 7
    best_k = 1
    
    primes_list = generate_primes(10**6)
    
    for p in primes_list:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
