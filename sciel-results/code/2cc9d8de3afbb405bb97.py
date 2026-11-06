"""Rewritten program using segmented sieve + Pollard-Brent verification."""
import math
import random
import time

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    """Pollard's Brent cycle detection factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, 10000)
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
        # retry with different parameters

def check_p_smooth_large(v, p):
    """
    Check if v is p-smooth, assuming v has no prime factors <= SMALL_PRIME_LIMIT.
    v is the remainder after dividing by small primes.
    """
    if v == 1:
        return True
    if v <= p:
        return True
    # v > p and v has no small prime factors
    if is_prime_mr(v):
        return v <= p
    # v is composite. Use Pollard-Brent
    def _has_large_factor(m, p_limit):
        if m <= 1:
            return False
        if m <= p_limit:
            return False
        if is_prime_mr(m):
            return m > p_limit
        d = pollard_brent(m)
        return _has_large_factor(d, p_limit) or _has_large_factor(m // d, p_limit)
    
    return not _has_large_factor(v, p)

def construct():
    best_k = 0
    best_p = 2
    
    # Generate primes up to a limit for the "segmented sieve" part
    SIEVE_LIMIT = 2 * 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    # Phase 1: Test primes up to SIEVE_LIMIT using the fast "remainder" method
    # For p <= SIEVE_LIMIT, we can divide by all primes q <= p.
    # If remainder is 1, it is p-smooth.
    
    for idx, p in enumerate(primes):
        if time.time() - start_time > time_limit:
            break
        
        # Optimization: if p is large, the number of primes to check is large.
        # But we can stop early if we find a failure.
        
        rems = [p * p + i for i in range(1, 201)]
        active = [i for i in range(200) if rems[i] > 1]
        
        # Iterate through primes q <= p
        # We only need primes up to p
        # Since p is from the primes list, we can find the index
        # But iterating all primes from 2 to p is efficient enough for p < 2e6?
        # pi(2e6) is ~148,000.
        # For each p, we do ~148,000 * (active count) ops.
        # Active count decreases quickly.
        
        for q in primes:
            if q > p:
                break
            if not active:
                break
            
            new_active = []
            for i in active:
                while rems[i] % q == 0:
                    rems[i] //= q
                if rems[i] > 1:
                    new_active.append(i)
            active = new_active
        
        # Now determine k
        # If p <= SIEVE_LIMIT:
        # rems[i] == 1 iff v_i is p-smooth
        # rems[i] > 1 iff v_i has a prime factor > p (since we divided all q <= p)
        k = 0
        for i in range(200):
            if rems[i] == 1:
                k = i + 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
        if k == 200:
            break

    # Phase 2: Test larger primes (p > SIEVE_LIMIT)
    # We sample primes in larger ranges.
    # For these, we divide by primes up to SIEVE_LIMIT.
    # Then check remainders with Pollard-Brent.
    
    if time.time() - start_time < time_limit:
        # Search in chunks
        # We need to generate primes in the range [SIEVE_LIMIT, 10^8] or so
        # But we can't sieve that large easily.
        # We can use Miller-Rabin to test candidates.
        
        # Let's test primes in [SIEVE_LIMIT, 10^7]
        # We'll step by 2 and check primality.
        
        p_start = SIEVE_LIMIT + 1
        if p_start % 2 == 0:
            p_start += 1
            
        p_end = 10**7
        
        for p in range(p_start, p_end, 2):
            if time.time() - start_time > time_limit:
                break
            if not is_prime_mr(p):
                continue
            
            rems = [p * p + i for i in range(1, 201)]
            
            # Divide by primes up to SIEVE_LIMIT
            for q in primes:
                # q is always <= SIEVE_LIMIT < p, so we divide
                # Optimization: only check if q*q <= min(rems)? No, we need to remove all small factors.
                # But if q > max(rems), we can stop? No, rems are ~ p^2.
                # We can stop if q > max(rems)? No.
                # We just iterate all primes in our list.
                
                # Early break if all rems are 1?
                # We can track active indices.
                
                active = [i for i in range(200) if rems[i] > 1]
                if not active:
                    break
                
                new_active = []
                for i in active:
                    while rems[i] % q == 0:
                        rems[i] //= q
                    if rems[i] > 1:
                        new_active.append(i)
                active = new_active
            
            # Now check rems
            k = 0
            for i in range(200):
                if rems[i] == 1:
                    k = i + 1
                else:
                    # rems[i] > 1, has no factors <= SIEVE_LIMIT
                    # Check if it is p-smooth
                    if check_p_smooth_large(rems[i], p):
                        k = i + 1
                    else:
                        break
            
            if k > best_k:
                best_k = k
                best_p = p
                
            if k == 200:
                break

    return {'p': best_p, 'k': best_k}
