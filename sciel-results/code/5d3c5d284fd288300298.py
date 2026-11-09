import time
import random
import math

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Witnesses for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in witnesses:
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

    # Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if is_prime_mr(n):
            return n
            
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
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
                r += 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def has_large_prime_factor(n, limit):
        """Check if n has a prime factor > limit."""
        if n <= limit:
            return False
        if is_prime_mr(n):
            return True
        d = pollard_brent(n)
        return has_large_prime_factor(d, limit) or has_large_prime_factor(n // d, limit)

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        return not has_large_prime_factor(n, p)

    def sieve_primes(start, end):
        """Simple sieve for a segment."""
        size = end - start
        # We need to handle even numbers carefully or just sieve odds
        # For simplicity and speed in Python, we'll sieve all and filter
        # But 10^6 size is fine.
        is_prime = bytearray(b'\x01') * (size + 1)
        is_prime[0] = 0
        if size >= 1:
            is_prime[1] = 0
            
        limit = int(math.isqrt(end))
        # We need primes up to sqrt(end) to sieve
        # Generating small primes once is better
        pass
        
    # Precompute small primes up to 10^5 for initial sieving of segments?
    # Actually, for a segment [L, R], we need to mark multiples of primes <= sqrt(R).
    # Let's generate primes up to 10000 once.
    small_primes = []
    limit_sp = 10000
    sp_sieve = bytearray(b'\x01') * (limit_sp + 1)
    sp_sieve[0] = sp_sieve[1] = 0
    for i in range(2, int(limit_sp**0.5) + 1):
        if sp_sieve[i]:
            for j in range(i*i, limit_sp + 1, i):
                sp_sieve[j] = 0
    small_primes = [i for i in range(2, limit_sp + 1) if sp_sieve[i]]

    def get_primes_in_segment(L, R):
        size = R - L + 1
        if size <= 0:
            return []
        is_prime = bytearray(b'\x01') * size
        if L == 0:
            is_prime[0] = 0
        if L == 1:
            is_prime[0] = 0
            if size > 1:
                is_prime[1] = 0
                
        limit = int(math.isqrt(R))
        for p in small_primes:
            if p > limit:
                break
            # Find first multiple of p in [L, R]
            start_idx = (L % p)
            if start_idx == 0:
                first_multiple = L
            else:
                first_multiple = L + (p - start_idx)
            # Mark multiples
            # Note: if first_multiple == p, we should not mark it as composite if p is in range
            # But p <= 10000 and L >= 10^7 usually, so p is not in range.
            # If L < p, we need to be careful.
            # For our ranges (10^7+), p is always < L.
            for j in range(first_multiple - L, size, p):
                is_prime[j] = 0
                
        primes = []
        for i in range(size):
            if is_prime[i]:
                primes.append(L + i)
        return primes

    best_k = 0
    best_p = 0
    
    # Search strategy:
    # Try a few large segments.
    # Segments of size 10^6.
    # Start from 10^8 down to 10^7.
    
    segment_size = 10**6
    # Start high
    start_range = 10**8
    end_range = 10**7
    
    # Iterate segments in descending order
    current_start = start_range - segment_size + 1
    while current_start >= end_range and time.time() < deadline:
        current_end = current_start + segment_size - 1
        
        primes = get_primes_in_segment(current_start, current_end)
        
        for p in primes:
            if time.time() > deadline:
                break
                
            k = 0
            # Check j from 1 to 200
            for j in range(1, 201):
                n = p * p + j
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
                    
            if k > best_k:
                best_k = k
                best_p = p
                # If we found a very good k, we might want to stop or continue?
                # Maximize k. Continue to find better.
                
        current_start -= segment_size

    return {'p': best_p, 'k': best_k}
