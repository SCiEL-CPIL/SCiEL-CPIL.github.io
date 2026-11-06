import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in MR_BASES:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
        if n % 11 == 0:
            return 11
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
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

    def factorize(n, limit):
        """
        Returns True if all prime factors of n are <= limit.
        """
        if n <= 1:
            return True
        
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n % p == 0:
                if p > limit:
                    return False
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= limit
                break
        
        if is_prime(n):
            return n <= limit
        
        # Pollard Rho
        d = pollard_brent(n)
        return factorize(d, limit) and factorize(n // d, limit)

    def check_run(p, max_k=200):
        """
        Returns k such that p^2+1 ... p^2+k are p-smooth.
        Stops at max_k or when a non-p-smooth number is found.
        """
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not factorize(n, p):
                break
            k += 1
        return k

    # Sieve of Eratosthenes to get primes up to a limit
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Segmented sieve to generate primes in [low, high]
    def segmented_sieve(low, high):
        if high < 2:
            return []
        low = max(low, 2)
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1
        
        # Get primes up to sqrt(high)
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve_primes(limit)
        
        # Segmented sieve for [low, high]
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        if low == 1:
            is_prime_seg[0] = 0
        if low == 2:
            is_prime_seg[0] = 1
        else:
            is_prime_seg[0] = 0
        
        # Mark even numbers (except 2)
        if low % 2 != 0:
            for i in range(1, size, 2):
                is_prime_seg[i] = 0
        
        # Mark multiples of small primes
        for p in small_primes:
            if p == 2:
                continue
            # Find the first multiple of p in [low, high]
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - low, size, p):
                is_prime_seg[j] = 0
        
        result = []
        for i in range(size):
            if is_prime_seg[i]:
                result.append(low + i)
        return result

    # Phase 1: Check small primes quickly to establish baseline
    best_k = 0
    best_p = 2
    
    # Check primes up to 10000
    small_limit = 10000
    primes_small = sieve_primes(small_limit)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Use segmented sieve to search larger prime ranges
    # We'll search in chunks of 100000 primes at a time
    chunk_size = 100000
    current_low = small_limit + 1
    
    while current_low < 10**7:
        if time.time() > deadline:
            break
        
        current_high = current_low + chunk_size - 1
        primes_chunk = segmented_sieve(current_low, current_high)
        
        for p in primes_chunk:
            if time.time() > deadline:
                break
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        current_low = current_high + 1

    # Phase 3: If time permits, try larger primes using random search
    if time.time() < deadline - 20:
        # Generate random large primes and check them
        # We'll try primes in the range [10^7, 10^9]
        for _ in range(500):
            if time.time() > deadline:
                break
            # Pick a random number and find the next prime
            x = random.randrange(10**7, 10**9)
            if x % 2 == 0:
                x += 1
            while not is_prime(x):
                x += 2
                if x > 10**9:
                    break
            if x > 10**9:
                continue
            k = check_run(x)
            if k > best_k:
                best_k = k
                best_p = x

    return {'p': best_p, 'k': best_k}
