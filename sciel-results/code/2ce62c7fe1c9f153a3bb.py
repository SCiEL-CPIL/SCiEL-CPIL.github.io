import math
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 110  # Safety margin for 120s limit

    # Miller-Rabin for determinism on 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        # Small primes for quick check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        
        # Write n-1 as d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    # Pollard's Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randrange(1, n), random.randrange(1, n), 128
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
        return g

    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    # Precompute small primes for trial division
    # 10^6 is a good balance. Sieve takes ~0.1s.
    small_primes_list = generate_primes(10**6)

    def is_p_smooth(n, p):
        """
        Returns True if all prime factors of n are <= p.
        """
        # Trial division by small primes
        for sp in small_primes_list:
            if sp > p:
                # If the trial division prime exceeds p, we stop.
                # Any remaining factor n is either prime or composite.
                # If n > 1, its smallest prime factor is > p.
                return n == 1
            
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return True
            
            # Early exit optimization:
            # If n is small, we can check primality directly
            if n < 10**12:
                if is_prime(n):
                    return n <= p
                # If n is composite and < 10^12, it has a factor < 10^6.
                # But we have already divided by all primes < 10^6 (since sp loop goes up to 10^6).
                # Wait, if p >= 10^6, we have divided by all primes <= 10^6.
                # If n < 10^12 and composite, it must be a product of two primes > 10^6?
                # No, if it had a factor <= 10^6, we would have divided it.
                # So if n < 10^12 and composite, it is a product of two primes > 10^6.
                # Let's say n = a * b with a, b > 10^6.
                # We need to check if a <= p and b <= p.
                # We can find a factor using Pollard-Brent.
                pass
        
        if n == 1:
            return False # Wait, if n became 1 in the loop, we returned True.
                         # If we exit loop with n > 1, we continue.
        
        # If we finished trial division without sp > p, 
        # it means we checked all primes up to 10^6.
        # If p < 10^6, we would have exited earlier with return n == 1.
        # So here p >= 10^6.
        
        # n might still have factors between 10^6 and p.
        # Or n might be prime.
        # Or n might be composite with large factors.
        
        if is_prime(n):
            return n <= p
            
        # n is composite.
        while n > 1:
            # Optimization: if n is small, trial divide? 
            # No, we already did small primes.
            f = pollard_brent(n)
            if f > p:
                return False
            n //= f
            if n == 1:
                return True
            if is_prime(n):
                return n <= p
            # If n is still composite, loop again
            
        return False

    def get_random_prime(min_val, max_val):
        """Generate a random prime in [min_val, max_val]"""
        while True:
            # Ensure odd
            n = random.randint(min_val, max_val)
            if n % 2 == 0:
                n += 1
            if n > max_val:
                continue
            if is_prime(n):
                return n

    best_p = 0
    best_k = 0

    # Phase 1: Sequential scan of small primes
    # Primes up to 10^6 are fast to check.
    # We already generated primes up to 10^6.
    # Let's scan the first 10000 primes or all primes up to 10^6?
    # 10^6 has ~78000 primes.
    # Checking each takes ~1-5 ms?
    # 78000 * 5ms = 390s. Too slow.
    # Let's scan up to 10^5 (9592 primes).
    # 9592 * 5ms = 48s. Still risky.
    # Let's scan up to 10^4 (1229 primes).
    # 1229 * 5ms = 6s. Safe.
    
    primes_to_scan = [p for p in small_primes_list if p <= 10**5]
    
    for p in primes_to_scan:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search in larger ranges
    ranges = [
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9)
    ]
    
    for min_val, max_val in ranges:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Try many attempts
        # We have ~60s left.
        # Each check is ~1ms for small k.
        # So we can do ~60000 checks?
        # Let's do 5000 attempts.
        for _ in range(5000):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            p = get_random_prime(min_val, max_val)
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > TIME_LIMIT:
                    break
                n = base + k + 1
                if not is_p_smooth(n, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p

    # Fallback
    if best_p == 0:
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
