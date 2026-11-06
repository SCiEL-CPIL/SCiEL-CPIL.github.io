import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # 10s margin
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        factors = []
        # Trial division by very small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard Rho for remaining
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run(p):
        """
        Returns the length of the run of p-smooth numbers starting from p^2+1.
        Max 200.
        """
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            # Quick check: if n is prime, it has a factor > p (itself)
            if is_prime(n):
                break
            
            # Check if all factors <= p
            # We don't need full factorization, just to see if there's a factor > p
            # But factorize is fine for n ~ p^2
            factors = factorize(n)
            if max(factors) > p:
                break
            k += 1
        return k

    # Generate primes up to 10^6 for sequential search
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Sequential scan of small primes
    for p in primes:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random sampling of larger primes
    # If we have time left, try random primes in [10^6, 10^9]
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        # Simple random prime generation
        # We can pick random odd numbers and check primality
        attempts = 0
        while time.time() - start_time < DEADLINE:
            # Pick random number in range
            # Use a range that is not too large to avoid bias, but large enough
            # Let's try 10^6 to 10^8
            candidate = random.randint(10**6, 10**8)
            if candidate % 2 == 0:
                candidate += 1
            # Check primality
            if is_prime(candidate):
                k = check_run(candidate)
                if k > best_k:
                    best_k = k
                    best_p = candidate
            attempts += 1
            if attempts > 10000: # Safety break
                break

    # Ensure valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
