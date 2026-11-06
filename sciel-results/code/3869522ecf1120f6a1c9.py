import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 135  # Leave 15s margin for 150s limit

    # Pre-compute small primes for trial division - use larger bound for better filtering
    LIMIT_SMALL = 10**6
    small_primes = []
    is_prime_small = bytearray([1]) * (LIMIT_SMALL + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(LIMIT_SMALL**0.5) + 1):
        if is_prime_small[i]:
            step = i
            start = i * i
            is_prime_small[start:LIMIT_SMALL+1:step] = bytearray(len(is_prime_small[start:LIMIT_SMALL+1:step]))
    for i in range(2, LIMIT_SMALL + 1):
        if is_prime_small[i]:
            small_primes.append(i)

    # Fast deterministic Miller-Rabin for n < 3.3e24
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
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

    def is_p_smooth(n, p):
        """Check if n has all prime factors <= p."""
        if n <= 1:
            return True
        if n <= p:
            return True
        # Quick check: if n is prime, fail
        if is_prime(n):
            return False  # n > p and prime, so factor > p
        
        # Trial division by small primes up to min(p, LIMIT_SMALL)
        remaining = n
        max_div = min(p, LIMIT_SMALL)
        for d in small_primes:
            if d > max_div:
                break
            if d * d > remaining:
                break
            while remaining % d == 0:
                remaining //= d
            if remaining == 1:
                return True
        
        if remaining == 1:
            return True
        if remaining <= p:
            return True
        
        # remaining > p, not divisible by any prime <= min(p, LIMIT_SMALL)
        # If remaining is prime, it's > p, so fail
        if is_prime(remaining):
            return False
        
        # remaining is composite with all prime factors > min(p, LIMIT_SMALL)
        # Since remaining > p, at least one prime factor > p
        # Find a factor to confirm
        f = pollard_brent(remaining)
        if f == remaining:
            # Shouldn't happen if composite
            return is_prime(remaining) and remaining <= p
        if f > p:
            # f is a factor > p, so n has a prime factor > p
            return False
        # f <= p, check if remaining/f is also p-smooth
        return is_p_smooth(remaining // f, p)

    best_p = 0
    best_k = 0
    
    # Structural change: Use segmented search over prime ranges
    # For each candidate prime p, first do a fast pre-filter on p^2+1
    # Only proceed to full scan if p^2+1 passes
    
    # Search in structured ranges, prioritizing mid-size primes
    # where p^2+j values are large enough to have many small factors
    # but p is large enough that "factor <= p" is less restrictive
    
    search_ranges = [
        (10**5, 5*10**5),
        (5*10**5, 2*10**6),
        (2*10**6, 10**7),
        (10**7, 5*10**7),
        (5*10**7, 2*10**8),
    ]
    
    rng = random.Random(42)
    
    for lo, hi in search_ranges:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Sample primes in this range
        num_samples = max(50, int((hi - lo) / 100000))
        num_samples = min(num_samples, 500)
        
        for _ in range(num_samples):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            p = rng.randint(lo, hi)
            # Find next prime
            while not is_prime(p):
                p += 1
                if p > 10**9:
                    break
            
            if p > 10**9:
                continue
            
            # Fast pre-filter: check p^2 + 1 first
            n1 = p * p + 1
            if not is_p_smooth(n1, p):
                continue
            
            # Full scan
            k = 1
            base_sq = p * p
            while k < 200:
                if time.time() - start_time > TIME_LIMIT:
                    break
                n = base_sq + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p

    # If we haven't found good results, do more random sampling
    if best_k < 10 and time.time() - start_time < TIME_LIMIT:
        for _ in range(200):
            if time.time() - start_time > TIME_LIMIT:
                break
            p = rng.randint(10**5, 5*10**8)
            while not is_prime(p):
                p += 1
                if p > 10**9:
                    break
            if p > 10**9:
                continue
            n1 = p * p + 1
            if not is_p_smooth(n1, p):
                continue
            k = 1
            base_sq = p * p
            while k < 200:
                if time.time() - start_time > TIME_LIMIT:
                    break
                n = base_sq + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    if best_p == 0:
        # Fallback: return a valid small result
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
