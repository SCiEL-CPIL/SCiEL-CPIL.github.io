import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = 5
        while d * d <= n:
            if n % d == 0 or n % (d + 2) == 0:
                return False
            d += 6
        return True
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p using trial division."""
        if n <= 1:
            return True
        # Try small primes first
        for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if q > p:
                break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        # Now n has no small prime factors. Check if n <= p (then it's prime <= p)
        if n <= p:
            return True
        # n > p and n has no prime factor <= min(p, 97). If n is prime, it's > p -> fail.
        # If n is composite, it has a prime factor <= sqrt(n). If that factor > p, fail.
        # If n is composite and all factors <= p, we need to verify.
        # Since n > p and we've removed all primes <= min(p,97), 
        # if p >= 97, then n has no prime factor <= p from our trial, so if n>1 and n<=p*p,
        # n could be product of two primes both in (97, p] or one prime > p.
        # Simplest: if n > p, check if n is prime -> if yes, fail (since n > p).
        # If n is composite, it has factor <= sqrt(n). If sqrt(n) <= p, we'd have found it.
        # So if n > p and n is not 1, and we've divided out all primes <= min(p, 97):
        #   - If p < 97: we've checked all primes <= p, so if n > 1, n has a prime factor > p -> fail
        #   - If p >= 97: n has no prime factor <= 97. If n <= p, we'd have returned. 
        #     n > p. If n is prime, n > p -> fail. If n is composite, smallest prime factor > 97.
        #     We need to check if all prime factors of n are <= p.
        #     Since n could be large, do a more careful check.
        
        # For now, conservative: if n > p, we need to factor n further.
        # Use trial division up to sqrt(n) but only need to check up to p.
        d = 101
        limit = int(math.isqrt(n))
        while d <= limit and d <= p:
            if n % d == 0:
                n //= d
                limit = int(math.isqrt(n))
                # Reset d to 2? No, we've already checked small ones. Continue from d.
                # Actually we need to recheck from 2 since n changed. But we know n has no factor <= 97.
                # Just continue trial division from d.
                d = 101  # restart from 101
                continue
            d += 2
        # After trial division up to min(sqrt(n), p), if n > 1:
        # if n <= p, it's a prime factor <= p -> ok
        # if n > p, it's a prime factor > p -> fail
        return n <= p
    
    def find_max_k(p, max_k=200):
        """Find the maximum k such that p^2+1..p^2+k are all p-smooth."""
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth(p2 + j, p):
                break
            k += 1
        return k
    
    # Known good candidates to check first
    known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    
    best_k = 0
    best_p = 2
    
    # Check known primes first
    for p in known_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Search for more primes in ranges
    # Try primes in various ranges
    search_ranges = [
        (1000, 5000),
        (5000, 10000),
        (10000, 50000),
        (50000, 100000),
        (100000, 500000),
        (500000, 1000000),
        (1000000, 5000000),
        (5000000, 10000000),
        (10000000, 50000000),
        (50000000, 100000000),
        (100000000, 500000000),
        (500000000, 1000000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        # Sample primes in this range
        step = max(1, (hi - lo) // 100)
        for p in range(lo, hi + 1, step):
            if time.time() - start_time > DEADLINE:
                break
            if p % 2 == 0 and p != 2:
                continue
            if p % 3 == 0 and p != 3:
                continue
            if not is_prime(p):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
                # Also check neighbors
                for dp in [-2, -4, -6, 2, 4, 6]:
                    np_ = p + dp
                    if np_ > 1 and np_ <= 10**9 and time.time() - start_time < DEADLINE:
                        if is_prime(np_):
                            nk = find_max_k(np_)
                            if nk > best_k:
                                best_k = nk
                                best_p = np_
    
    # Do a denser search around the best found
    if best_p > 2 and time.time() - start_time < DEADLINE:
        center = best_p
        for offset in range(-1000, 1001, 2):
            if time.time() - start_time > DEADLINE:
                break
            p = center + offset
            if p < 2 or p > 10**9:
                continue
            if p % 2 == 0 and p != 2:
                continue
            if p % 3 == 0 and p != 3:
                continue
            if not is_prime(p):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
