import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # 2. Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
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
            if g != n:
                return g

    # 3. Generate primes up to 10^6
    limit = 10**6
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime_flag in enumerate(sieve) if is_prime_flag]
    small_primes = primes  # All primes up to 10^6

    # 4. Helper: Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # Divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if n <= p:
                    return True
        
        if n == 1:
            return True
        if n <= p:
            return True
            
        # If we get here, n has no prime factors <= min(p, 10^6)
        # And n > p
        
        # If p < 10^6, we have divided by all primes <= p.
        # So n must be prime.
        if p < 10**6:
            return is_prime(n) and n <= p # n > p, so this is False
        
        # If p >= 10^6:
        # n has no prime factors <= 10^6.
        # n > p >= 10^6.
        
        # Check if n is prime
        if is_prime(n):
            return n <= p # n > p, so False
        
        # n is composite.
        # Find a factor
        f = pollard_brent(n)
        if f == n:
            # Should not happen if is_prime is correct and n is composite
            return False
            
        # Check if f is <= p
        if f > p:
            # f is a factor > p. 
            # If f is prime, n is not p-smooth.
            # If f is composite, it has a prime factor >= f? No.
            # But if f > p, and f is the smallest factor? 
            # pollard_brent doesn't guarantee smallest.
            # We need to check if n has ANY factor > p.
            # If f > p, it's possible n = f * g where g < p.
            # But we already removed all factors <= 10^6.
            # So if f > p >= 10^6, then f has no factors <= 10^6.
            # If f is prime, then f > p is a prime factor > p. Fail.
            # If f is composite, let g be smallest prime factor of f.
            # g > 10^6.
            # If g > p, Fail.
            # If g <= p, then f is p-smooth? No, we need to check the rest.
            
            # Simpler: Just check if the largest prime factor of n is <= p.
            # We can recursively check factors.
            
            def check_m(m):
                if m == 1:
                    return True
                if m <= p:
                    return True
                if is_prime(m):
                    return m <= p
                ff = pollard_brent(m)
                if ff == m:
                    return False
                return check_m(ff) and check_m(m // ff)
                
            return check_m(n)
        else:
            # f <= p
            # n = f * (n//f)
            # Since f <= p, we just need to check if n//f is p-smooth.
            # n//f has no factors <= 10^6 (since n did, and f did? 
            # Wait, f might have small factors? No, n had no small factors, so f doesn't.
            # So n//f has no factors <= 10^6.
            return is_p_smooth(n // f, p)

    # Main search loop
    best_k = 0
    best_p = 2
    
    # Phase 1: Scan primes up to 10^5
    for p in primes:
        if time.time() > deadline:
            break
        if p > 100000:
            break
            
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search for larger primes
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        lower = 100000
        upper = 10**9
        
        p_candidate = random.randint(lower, upper)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        while not is_prime(p_candidate):
            p_candidate += 2
            if p_candidate > upper:
                break
                
        if p_candidate > upper:
            continue
            
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p_candidate * p_candidate + k + 1
            if is_p_smooth(n, p_candidate):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p_candidate

    return {'p': best_p, 'k': best_k}
