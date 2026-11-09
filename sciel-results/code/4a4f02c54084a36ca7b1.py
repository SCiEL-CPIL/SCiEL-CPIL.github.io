import time
import math
import random

def construct():
    deadline = time.time() + 145  # safety margin before 150s
    
    # Small primes for trial division
    small_primes = []
    d = 2
    while d <= 100000:
        is_p = True
        for q in small_primes:
            if q * q > d:
                break
            if d % q == 0:
                is_p = False
                break
        if is_p:
            small_primes.append(d)
        d += 1 if d == 2 else 2
    
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def segmented_sieve_smooth(p, k=200):
        """
        For prime p, compute for each offset i in [1, k] whether p^2+i is p-smooth.
        Returns list of booleans where result[i-1] is True if p^2+i is p-smooth.
        Uses segmented sieve with all primes <= p.
        """
        p2 = p * p
        # We need to check p^2+1 through p^2+k
        # A number n is p-smooth iff all its prime factors are <= p
        # Equivalently, if we divide out all prime factors <= p, the remainder must be 1
        
        # Strategy: for each prime q <= p, mark multiples of q in [p^2+1, p^2+k]
        # But we need to know if the ENTIRE number is composed of primes <= p.
        # 
        # Better approach: For each number n in [p^2+1, p^2+k], try to factor it
        # by dividing by all primes <= p. If we can reduce it to 1, it's p-smooth.
        #
        # Even better: use a segmented sieve approach.
        # Create an array rem[i] = p^2 + i for i in 1..k
        # For each prime q <= p, for each multiple of q in [p^2+1, p^2+k], divide rem by q.
        # At the end, rem[i] == 1 means p-smooth.
        
        rem = [p2 + i for i in range(1, k + 1)]
        
        # Sieve with all primes <= p
        # For each prime q <= p, find the first multiple of q in [p^2+1, p^2+k]
        # and mark/divide all such positions.
        
        for q in small_primes:
            if q > p:
                break
            # Find first index i in [1,k] such that (p2+i) % q == 0
            # p2 + i ≡ 0 (mod q) => i ≡ -p2 (mod q)
            # i_start = ((-p2) % q + 1) if (-p2) % q != 0 else 1... let me think more carefully
            # We need i in [1, k] such that (p2 + i) % q == 0
            # i ≡ (-p2) mod q
            r = (-p2) % q
            if r == 0:
                i_start = 1  # i=1: p2+1, check if (p2+1)%q==0... wait
                # Actually i ≡ 0 mod q means i can be q, 2q, ... but also i=0 which is out of range
                # Let me reconsider: we need i such that (p2+i) % q == 0
                # i ≡ (-p2) mod q
                # If (-p2) mod q == 0, then i ≡ 0 mod q, so i = q, 2q, ...
                # But i must be >= 1, so first valid i is q (if q <= k)
                i_start = q
            else:
                i_start = r  # r is in [1, q-1], so i_start = r
            # But wait, if r == 0, i_start should be q (since i must be >= 1 and i ≡ 0 mod q)
            # If r != 0, i_start = r (which is in [1, q-1])
            
            if i_start > k:
                continue
            
            # Now for i = i_start, i_start+q, i_start+2q, ... <= k, divide rem[i-1] by q
            for idx in range(i_start - 1, k, q):
                rem[idx] //= q
        
        # Check which rem[i] are 1
        result = [r == 1 for r in rem]
        return result
    
    def compute_k(p, max_k=200):
        """Compute the run length k for prime p using segmented sieve."""
        if p <= 100000:
            # Use segmented sieve - efficient for small p
            smooth = segmented_sieve_smooth(p, max_k)
            k = 0
            for s in smooth:
                if s:
                    k += 1
                else:
                    break
            return k
        else:
            # For large p, segmented sieve with all primes up to p is expensive
            # Fall back to per-number checking
            k = 0
            p2 = p * p
            while k < max_k:
                n = p2 + k + 1
                temp = n
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > temp:
                        break
                    while temp % sp == 0:
                        temp //= sp
                if temp == 1:
                    k += 1
                    continue
                if temp <= p:
                    k += 1
                    continue
                if is_prime_mr(temp):
                    if temp <= p:
                        k += 1
                    else:
                        break
                    continue
                # Need full factorization - use trial division up to sqrt(temp)
                # This is slow but rare
                d = sp + 1 if sp < 100000 else 100001
                d |= 1  # make odd
                while d * d <= temp:
                    if temp % d == 0:
                        if d > p:
                            break
                        while temp % d == 0:
                            temp //= d
                        if d > p:
                            break
                    d += 2
                else:
                    if temp == 1 or temp <= p:
                        k += 1
                        continue
                    else:
                        break
                break
            return k
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to 10^6 for initial search
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes_small = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Try small primes first with segmented sieve (fast)
    for p in primes_small:
        if time.time() > deadline:
            break
        k = compute_k(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
    
    # Try larger primes
    if time.time() < deadline - 5:
        # Try primes in ranges, using the slower per-number method
        # Focus on ranges where long runs might occur
        # Heuristic: try primes where p^2 mod small_primes gives favorable residues
        
        # Try some structured search: primes near multiples of small primes
        # Also try random primes in larger ranges
        
        # First, try primes from 10^6 to 10^7
        if time.time() < deadline - 10:
            # Generate primes in [10^6, 10^7] using segmented sieve
            lo, hi = 10**6, 10**7
            seg_size = hi - lo + 1
            seg_sieve = bytearray(b'\x01') * seg_size
            for q in small_primes:
                if q * q > hi:
                    break
                start = ((lo + q - 1) // q) * q
                if start < q * q:
                    start = q * q
                for j in range(start - lo, seg_size, q):
                    seg_sieve[j] = 0
            primes_mid = [lo + i for i in range(seg_size) if seg_sieve[i]]
            
            for p in primes_mid:
                if time.time() > deadline:
                    break
                k = compute_k(p)
                if k > best_k:
                    best_k = k
                    best_p = p
                if best_k >= 200:
                    break
        
        # Try random larger primes
        if time.time() < deadline - 5:
            random.seed(42)
            for _ in range(500):
                if time.time() > deadline:
                    break
                p = random.randint(10**7, 10**9)
                if p % 2 == 0:
                    p += 1
                while not is_prime_mr(p):
                    p += 2
                    if p > 10**9:
                        break
                if p > 10**9:
                    break
                k = compute_k(p)
                if k > best_k:
                    best_k = k
                    best_p = p
                if best_k >= 200:
                    break
    
    return {'p': best_p, 'k': best_k}
