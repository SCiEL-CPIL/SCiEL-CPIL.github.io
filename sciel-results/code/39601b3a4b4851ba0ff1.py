import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in small_primes:
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

    # Sieve primes up to limit
    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Segmented sieve to find primes in [low, high]
    def segmented_sieve(low, high):
        if low > high:
            return []
        if low == 1:
            low = 2
        size = high - low + 1
        is_prime_seg = bytearray([1]) * size
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve(limit)
        for q in small_primes:
            if q * q > high:
                break
            start = max(q * q, ((low + q - 1) // q) * q)
            for j in range(start - low, size, q):
                is_prime_seg[j] = 0
        primes = []
        for i in range(size):
            if is_prime_seg[i]:
                primes.append(low + i)
        return primes

    # Check run length for a given prime p using segmented factorization
    # For each i in 1..max_k, check if p^2+i is p-smooth
    # Key insight: p^2+i has a prime factor > p iff it has a prime factor in (p, p^2+i]
    # Since p^2+i < (p+1)^2 for i < 2p+1, and for i <= 200 with p >= 200, p^2+i < (p+1)^2
    # So any prime factor > p of p^2+i must be > p and <= p^2+i
    # But we can check: after removing all prime factors <= p, if remainder > 1, it has a factor > p
    # More efficiently: p^2+i is p-smooth iff for every prime q <= p, we can divide it out
    # But that's expensive. Instead, note that if p^2+i has a prime factor r > p, then r >= smallest prime > p
    # Actually, the simplest check: factor p^2+i completely and verify all factors <= p
    # For the segmented approach: precompute which primes q <= p divide which of p^2+1, ..., p^2+200
    def check_run_length_fast(p, max_k=200, primes_list=None):
        """
        Check the maximum k such that p^2+1, ..., p^2+k are all p-smooth.
        Uses the fact that for each prime q <= p, we can find all i in [1,max_k] 
        where q divides p^2+i, and divide out all powers of q from those positions.
        """
        if primes_list is None:
            # Need primes up to p
            if p <= 1000000:
                primes_list = sieve(p)
            else:
                primes_list = sieve(min(p, 500000))
        
        p2 = p * p
        # rem[i] will hold the remaining part of p^2+i after removing all prime factors <= p
        rem = [1] * (max_k + 1)
        for i in range(1, max_k + 1):
            rem[i] = p2 + i
        
        # For each prime q <= p, find positions i where q | (p^2 + i)
        # p^2 + i ≡ 0 (mod q) => i ≡ -p^2 (mod q)
        for q in primes_list:
            if q > p:
                break
            # Find i in [1, max_k] such that i ≡ (-p^2) mod q
            r = (-p2) % q
            if r == 0:
                r = q
            # i = r, r+q, r+2q, ... <= max_k
            i = r
            while i <= max_k:
                # Divide out all powers of q from rem[i]
                while rem[i] % q == 0:
                    rem[i] //= q
                i += q
        
        # Count consecutive positions starting from 1 where rem[i] == 1
        k = 0
        for i in range(1, max_k + 1):
            if rem[i] == 1:
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive scan of small primes using fast check
    # For p <= 10000, we can sieve primes up to p cheaply
    small_primes = sieve(10000)
    for p in small_primes:
        if time.time() - start_time > deadline - 60:
            break
        k = check_run_length_fast(p, max_k=200, primes_list=small_primes)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Segmented sieve scan for larger primes
    # Use segmented sieve to generate primes in ranges, and for each prime,
    # use the fast check_run_length with pre-sieved primes up to min(p, limit)
    
    # Pre-sieve primes up to 100000 for use in factoring
    factoring_primes = sieve(100000)
    
    # Scan primes in segments from 10000 to as far as time allows
    segment_size = 100000
    current_low = 10000
    
    while time.time() - start_time < deadline - 30:
        current_high = min(current_low + segment_size - 1, 10**9)
        seg_primes = segmented_sieve(current_low, current_high)
        
        for p in seg_primes:
            if time.time() - start_time > deadline - 25:
                break
            # For the fast check, we need primes up to min(p, len(factoring_primes)-1)
            # If p > 100000, we only have primes up to 100000, which may not be enough
            # But for p > 100000, p^2+i for i <= 200 has sqrt ~ p, so we need primes up to ~p
            # However, if p > 100000, we can still check: if after removing primes <= 100000,
            # the remainder is 1, it's p-smooth. If remainder > 1, it could be a prime in (100000, p]
            # or a composite. But if p > 100000 and remainder <= p, it's fine.
            # Actually, for p > 100000, if rem[i] > 1 after removing primes <= 100000:
            #   - If rem[i] <= p, it's p-smooth (all its prime factors are <= rem[i] <= p)
            #   - If rem[i] > p, it's NOT p-smooth
            # So we can still use factoring_primes (up to 100000) and just check if remainder <= p
            
            if p <= 100000:
                k = check_run_length_fast(p, max_k=200, primes_list=factoring_primes)
            else:
                # Custom check for p > 100000
                p2 = p * p
                max_k_test = 200
                rem = [p2 + i for i in range(1, max_k_test + 1)]
                for q in factoring_primes:
                    r = (-p2) % q
                    if r == 0:
                        r = q
                    i = r
                    while i <= max_k_test:
                        while rem[i-1] % q == 0:
                            rem[i-1] //= q
                        i += q
                # Now check: rem[i-1] should be 1 or <= p
                k = 0
                for i in range(1, max_k_test + 1):
                    if rem[i-1] == 1 or rem[i-1] <= p:
                        k += 1
                    else:
                        break
                # But wait: if rem[i-1] <= p but > 1, we need to verify it's actually p-smooth
                # Since rem[i-1] <= p, all its prime factors are <= rem[i-1] <= p, so it IS p-smooth
                # This is correct!
            
            if k > best_k:
                best_k = k
                best_p = p
        
        current_low = current_high + 1
        if current_low > 10**9:
            break

    # Phase 3: If time remains, try to extend the search or verify
    # Verify the best result found
    if best_k > 0:
        if not is_prime(best_p):
            best_p = 2
            best_k = 1
        
        # Full verification
        p2 = best_p * best_p
        # Sieve primes up to min(best_p, 200000) for verification
        verify_limit = min(best_p, 200000)
        verify_primes = sieve(verify_limit)
        
        valid = True
        for i in range(1, best_k + 1):
            n = p2 + i
            for q in verify_primes:
                if q > best_p:
                    break
                while n % q == 0:
                    n //= q
            if n > 1 and n > best_p:
                valid = False
                break
            # If n <= best_p, it's fine (all factors of n are <= n <= best_p)
        
        if not valid:
            # Binary search for actual valid k
            lo, hi = 1, best_k
            while lo < hi:
                mid = (lo + hi + 1) // 2
                valid_mid = True
                for i in range(1, mid + 1):
                    n = p2 + i
                    for q in verify_primes:
                        if q > best_p:
                            break
                        while n % q == 0:
                            n //= q
                    if n > 1 and n > best_p:
                        valid_mid = False
                        break
                if valid_mid:
                    lo = mid
                else:
                    hi = mid - 1
            best_k = lo

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
