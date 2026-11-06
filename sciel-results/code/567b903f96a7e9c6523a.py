import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        # Deterministic Miller-Rabin for 64-bit
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
                x = (x * x) % n
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

    # Pre-compute small primes
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    def is_p_smooth(n, p):
        """Check if n is p-smooth"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division
        temp_n = n
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > temp_n:
                break
            if temp_n % sp == 0:
                while temp_n % sp == 0:
                    temp_n //= sp
                if temp_n == 1:
                    return True
        
        if temp_n == 1:
            return True
            
        if is_prime(temp_n):
            return temp_n <= p
            
        # temp_n is composite
        # Find factors
        while temp_n > 1:
            if is_prime(temp_n):
                if temp_n > p:
                    return False
                break
            d = pollard_brent(temp_n)
            if d == temp_n:
                # Should not happen if is_prime is correct, but just in case
                continue
            if d > p:
                return False
            temp_n //= d
            while temp_n % d == 0:
                temp_n //= d
            if temp_n == 1:
                break
            if is_prime(temp_n):
                if temp_n > p:
                    return False
                break
        return True

    best_k = 0
    best_p = 2
    
    # Search strategy:
    # Check primes in increasing order of magnitude?
    # Or random?
    # Let's try a mixed approach.
    # First, check a range of smaller primes where checking is fast.
    # Then random large primes.
    
    # Generate primes in [10^5, 10^6]
    # We can just iterate odd numbers and check primality.
    # Or use a segmented sieve.
    # Given time, let's just iterate and check.
    
    # To cover more ground, let's iterate p from 10^5 to 10^7 in steps?
    # No, we need p to be prime.
    
    # Let's try to check all primes in [10^5, 10^6] first.
    # 78k primes.
    # For each, check k=1.
    # If pass, check k=2, etc.
    
    # Optimization:
    # If p is small, p^2 is small.
    # is_p_smooth is fast.
    
    current_p = 100000
    while time.time() - start_time < DEADLINE:
        # Find next prime
        if current_p % 2 == 0:
            current_p += 1
        else:
            current_p += 2
            
        if is_prime(current_p):
            p = current_p
            p2 = p * p
            k = 0
            # Check run
            # Limit k to 200
            while k < 200:
                val = p2 + k + 1
                if not is_p_smooth(val, p):
                    break
                k += 1
                # Time check inside inner loop?
                if time.time() - start_time > DEADLINE:
                    break
                    
            if k > best_k:
                best_k = k
                best_p = p
                # If we find a very high k, we might want to continue searching nearby?
                # Or just keep going.
        
        # If we've gone through 10^6 to 10^7, switch to random?
        # Let's just keep scanning up to 10^7.
        # If we reach 10^7, we can switch to random sampling in [10^7, 10^9].
        if current_p > 10000000:
             # Switch to random search
             # Pick random primes in [10^7, 10^9]
             for _ in range(100):
                 if time.time() - start_time > DEADLINE:
                     break
                 p = random.randint(10**7, 10**9)
                 p |= 1
                 if is_prime(p):
                     p2 = p * p
                     k = 0
                     while k < 200:
                         val = p2 + k + 1
                         if not is_p_smooth(val, p):
                             break
                         k += 1
                         if time.time() - start_time > DEADLINE:
                             break
                     if k > best_k:
                         best_k = k
                         best_p = p
             # After random search, break main loop?
             # Or continue scanning?
             # Let's break and return best.
             break
             
    return {'p': best_p, 'k': best_k}
