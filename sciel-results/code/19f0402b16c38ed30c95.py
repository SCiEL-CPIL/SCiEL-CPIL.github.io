import math
import time
import random

def construct():
    # Time limit guard
    start_time = time.time()
    TIME_LIMIT = 115  # 5 seconds safety margin
    
    # Miller-Rabin for determinism on 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Write n-1 as d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        # See https://miller-rabin.appspot.com/
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

    # Pollard's Brent's cycle detection
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

    # Precompute small primes for trial division
    def generate_small_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    small_primes = generate_small_primes(1000)

    def has_large_factor(n, p):
        """
        Returns True if n has any prime factor > p.
        Returns False if all prime factors are <= p.
        """
        # Trial division by small primes
        for d in small_primes:
            if d > p:
                break
            while n % d == 0:
                n //= d
        
        if n == 1:
            return False
            
        if is_prime(n):
            return n > p
            
        # n is composite
        # Find a factor
        while n > 1:
            if is_prime(n):
                return n > p
            
            # Try to find a factor
            f = pollard_brent(n)
            
            if f > p:
                return True
            
            # Divide out f
            n //= f
            
        return False

    def get_random_prime(low, high):
        while True:
            n = random.randrange(low, high)
            if n % 2 == 0:
                n += 1
            if is_prime(n):
                return n

    best_p = 7
    best_k = 1
    
    # Generate candidates
    candidates = []
    
    # Large primes first
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT:
            break
        candidates.append(get_random_prime(10**8, 10**9))
        
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT:
            break
        candidates.append(get_random_prime(10**6, 10**8))
        
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT:
            break
        candidates.append(get_random_prime(10**4, 10**6))
        
    # Sort by p descending
    candidates.sort(reverse=True)
    
    for p in candidates:
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
