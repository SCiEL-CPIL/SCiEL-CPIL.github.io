import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute small primes for trial division
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    small_primes = get_primes(10**6)
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2: return False
        # Small primes check
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Miller-Rabin
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases for n < 3,317,044,064,679,887,385,961,981
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

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
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
        # Trial division by small primes
        for q in small_primes:
            if q > p: break
            if q * q > n: break
            while n % q == 0:
                n //= q
        if n == 1: return True
        # If n > 1, check if it is prime
        if is_prime(n):
            return n <= p
        # If composite, factorize and check
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

    best_k = 0
    best_p = 2
    
    # Search primes
    # We'll iterate through potential primes p
    # To avoid checking non-primes, we can use a simple primality test
    # But generating primes is better.
    # Let's iterate p from 2 upwards, checking primality.
    # To speed up, we can jump.
    
    # Helper to check if p is prime (using small primes first)
    def is_prime_fast(n):
        if n < 2: return False
        for q in small_primes:
            if q * q > n: return True
            if n % q == 0: return n == q
        # If not divisible by small primes, do Miller Rabin
        return is_prime(n)

    # We will search in chunks to allow deadline checks
    # Start with small p, then larger
    # Note: p must be prime.
    
    p = 2
    while p <= 10**9:
        if time.time() > deadline:
            break
            
        if is_prime_fast(p):
            # Check k
            k = 0
            # We want to maximize k.
            # Check p^2+1 ... p^2+200
            # If any fails, stop.
            for j in range(1, 201):
                n = p * p + j
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                # If we find a very good k, we might want to continue to see if we can do better,
                # but usually k is small.
        
        p += 1 if p == 2 else 2 # Skip even numbers after 2

    return {'p': best_p, 'k': best_k}
