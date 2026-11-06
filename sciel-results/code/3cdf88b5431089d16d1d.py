import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

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
        # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
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

    # 2. Generate primes up to 10^7 using a simple sieve
    limit = 10**7
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime_flag in enumerate(sieve) if is_prime_flag]
    
    # Helper: Check if n is p-smooth using trial division by primes <= p
    # Optimization: If n is prime, it is p-smooth iff n <= p.
    # If n is composite, we divide by primes q <= p. If remainder is 1, smooth.
    # If remainder > 1, it has a prime factor > p (since we divided all <= p), so not smooth.
    def is_p_smooth(n, p, primes_list):
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # If n is prime, check if n <= p
        if is_prime(n):
            return n <= p
            
        # Trial division by primes q <= p
        # We can break if q*q > n, but we must also respect q <= p
        # Since n ~ p^2, q will go up to ~p
        for q in primes_list:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                # If n is now prime, check if <= p
                if is_prime(n):
                    return n <= p
                # If n is composite, we continue dividing
                # Note: if n < q*q, then n is prime (handled above) or 1
        
        # If we exit the loop, n has no prime factors <= p (and <= sqrt(n))
        # If n == 1, smooth
        if n == 1:
            return True
        # If n > 1, it has a prime factor > p (or n itself is prime > p)
        # Actually, if q*q > n and n > 1, n is prime.
        # If q > p and n > 1, n has factors > p.
        # In all cases where n > 1 after dividing all q <= min(p, sqrt(n)),
        # n is not p-smooth because any prime factor of n is > p.
        return False

    best_k = 0
    best_p = 0
    
    # Main search loop
    for p in primes:
        if time.time() > deadline:
            break
            
        if p < 2:
            continue
            
        # Quick check for p^2+1
        n1 = p * p + 1
        if not is_p_smooth(n1, p, primes):
            continue
            
        # If p^2+1 is smooth, check further
        k = 1
        for i in range(2, 201):
            if time.time() > deadline:
                break
            n = p * p + i
            if not is_p_smooth(n, p, primes):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
