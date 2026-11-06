import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    # Deterministic Miller-Rabin for 64-bit integers
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        # Small primes check
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        
        # Write n-1 as d * 2^s
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

    # Pollard's Rho algorithm with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
        g, q, r, x, ys = 1, 1, 1, 0, y
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

    def factorize(n):
        factors = []
        # Trial division by small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard's Rho for the remaining part
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            if d == m:
                # Retry with different parameters if it fails
                stack.append(m)
                continue
            stack.append(d)
            stack.append(m // d)
        return factors

    def is_p_smooth(n, p):
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        factors = factorize(n)
        return max(factors) <= p

    best_k = 0
    best_p = 2
    
    # Phase 1: Check primes up to 10^6
    limit1 = 10**6
    sieve = bytearray(b'\x01') * (limit1 + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit1 ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    for p in primes:
        if time.time() > deadline:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Check random primes up to 10^9
    # Use a generator of random odd numbers
    while time.time() < deadline:
        # Generate a random candidate in [10^6, 10^9]
        # Bias towards larger numbers? Or uniform?
        # Uniform in log space is better for prime density?
        # Prime density is 1/ln(x).
        # Random uniform in [1, 10^9] will miss large primes?
        # No, probability of being prime is 1/ln(x).
        # Let's just pick random odd integers.
        
        # Try to find a prime
        found_prime = False
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick random number between 10^6 and 10^9
            # Using random.randint
            cand = random.randint(10**6, 10**9)
            if cand % 2 == 0:
                cand += 1
            if is_prime(cand):
                p = cand
                found_prime = True
                break
        
        if not found_prime:
            continue
            
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
