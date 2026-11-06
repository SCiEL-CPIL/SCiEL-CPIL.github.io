import random
import math
import time

def construct():
    # 1. Sieve of Eratosthenes for small primes
    def get_primes(n):
        sieve = bytearray([True]) * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(10**5)
    small_primes = [p for p in primes if p < 10000]

    # 2. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
            m = 128
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
                        q = (q * abs(x - y)) % n
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

    # 4. Find largest prime factor
    def max_prime_factor(n, small_primes):
        if n <= 1:
            return 1
        
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                # Divide out all p
                while n % p == 0:
                    n //= p
                # If n becomes 1, p was the largest (or only)
                # But we need the max of p and factors of remaining n
                # Actually, if we divide out p, the remaining n has no factor p.
                # The largest factor is max(p, max_prime_factor(remaining_n))
                # But wait, if n becomes 1, the largest factor was p (if p was the largest prime factor so far)
                # However, there might be larger prime factors in the original n that were not divided yet?
                # No, we are iterating primes in ascending order.
                # So any remaining factor is > p.
                # So we just need to recurse on n.
                # But we must keep track of p.
                # Let's restructure:
                # Find all prime factors, return max.
                pass
        
        # Let's implement a recursive helper that returns the set of prime factors or just the max.
        # To be efficient, we just want the max.
        
        def _max_pf(m):
            if m == 1:
                return 1
            # Check small primes again? No, we can pass the list or just do it outside.
            # Let's do trial division in the main function first to reduce m.
            
            if is_prime(m):
                return m
            
            d = pollard_rho(m)
            return max(_max_pf(d), _max_pf(m // d))
        
        # First, strip small primes
        max_f = 1
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                max_f = max(max_f, p)
                while n % p == 0:
                    n //= p
        
        if n > 1:
            max_f = max(max_f, _max_pf(n))
            
        return max_f

    # 5. Search
    best_k = 0
    best_p = 2
    start_time = time.time()
    timeout = 115 # 5s buffer

    # Fast p-smooth check
    def is_p_smooth(n, p, primes_list):
        # Check if all prime factors of n are <= p
        # primes_list is sorted
        for q in primes_list:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        return n == 1

    # Exhaustive search for small primes
    # We limit to primes < 100000 to ensure speed
    # Pruning: if k=1 fails, we stop
    for p in primes:
        if p > 100000:
            break
        if time.time() - start_time > timeout:
            break
            
        k = 0
        for i in range(1, 201):
            n = p*p + i
            if not is_p_smooth(n, p, primes):
                break
            k += 1
            if k >= best_k:
                # Only update if strictly better or equal but we want max
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k == 200:
                        return {'p': best_p, 'k': best_k}
        
    # If we have time, do random search for larger primes
    # But random search is very unlikely to beat small primes for this problem
    # However, we include it just in case
    while time.time() - start_time < timeout:
        # Generate random prime
        while True:
            p = random.randrange(10**6, 10**9, 2) | 1
            if is_prime(p):
                break
        
        k = 0
        for i in range(1, 201):
            n = p*p + i
            if not is_p_smooth(n, p, primes):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
