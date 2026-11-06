import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Helper to check if a number is prime (for the remainder)
    # We already have primes up to 10^6, so we can use trial division
    def is_prime_large(n):
        if n < 2:
            return False
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                return n == p
        return True
        
    def get_factor(n):
        # Find a prime factor of n, assuming n is composite and > 1
        # and has no factors <= 10^4 (if we used that cutoff)
        # But to be safe, we just trial divide by all primes
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                return p
        return n # n is prime
        
    # 2. Segmented Phi Calculation
    def compute_phi_segment(L, W):
        # L is the start, W is the width
        vals = np.arange(L, L + W, dtype=np.int64)
        phi = vals.copy()
        
        # Sieve with small primes
        # We use primes up to 10^4 for the vectorized part
        # Primes up to 10^4 are the first ~1229 primes
        small_primes = [p for p in primes if p <= 10000]
        
        for p in small_primes:
            # Find first multiple of p in [L, L+W-1]
            start = L - (L % p)
            if start < L:
                start += p
            
            if start >= L + W:
                continue
                
            # Indices in the array
            idxs = np.arange(start, L + W, p, dtype=np.int64) - L
            
            # Check which are actually divisible (they should be, but vals might have been modified? No, vals is original L+i)
            # Wait, we are modifying vals.
            # We need to check if vals[idxs] is divisible by p.
            # Since we divide vals by p in previous steps, vals[idxs] might not be divisible by p anymore if p was already processed?
            # No, we process each p once.
            # But if a number is divisible by p^2, we need to divide it out completely.
            
            # Check divisibility in the current state of vals
            mask = (vals[idxs] % p == 0)
            if not np.any(mask):
                continue
                
            valid_idxs = idxs[mask]
            
            # Update phi
            phi[valid_idxs] -= phi[valid_idxs] // p
            
            # Divide vals by p as long as possible
            # This is a loop, but vectorized inside
            while np.any(vals[valid_idxs] % p == 0):
                vals[valid_idxs] //= p
                
        # Process remainders
        for i in range(W):
            r = vals[i]
            if r > 1:
                if is_prime_large(r):
                    phi[i] -= phi[i] // r
                else:
                    # Find a factor
                    q = get_factor(r)
                    if q != r:
                        phi[i] -= phi[i] // q
                        r //= q
                        if r > 1 and r != q:
                            phi[i] -= phi[i] // r
                        # If r == q, it was q^2, handled by first subtraction?
                        # phi(n) = n * (1-1/q) * (1-1/q) if q^2 | n?
                        # No, phi(n) = n * prod(1-1/p).
                        # If r = q^2, then n has factor q.
                        # We already subtracted phi//q.
                        # Do we need to subtract again?
                        # No, the factor is just q.
                        # phi(n) = n * (1-1/q).
                        # The (1-1/q) term is applied once for the prime q.
                        # So we are done.
                    else:
                        # r is prime, handled by is_prime_large?
                        # get_factor returns n if n is prime.
                        # So if q == r, it means r is prime.
                        # But is_prime_large said False?
                        # This can happen if r is a prime > 10^6?
                        # My is_prime_large uses primes up to 10^6.
                        # If r > 10^12, it's not possible.
                        # If r is prime > 10^6, is_prime_large will return True.
                        # So this case shouldn't happen if is_prime_large is correct.
                        pass

        return phi

    # 3. Find longest distinct run in an array
    def find_longest_distinct(arr):
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(len(arr)):
            val = arr[right]
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            if right - left + 1 > best_k:
                best_k = right - left + 1
                best_s = left
        return best_s, best_k

    best_k = 1
    best_s = 1
    max_n = 10**12
    W = 100000  # Segment width
    
    # Initial scan
    phi_arr = compute_phi_segment(1, W)
    s, k = find_longest_distinct(phi_arr)
    if k > best_k:
        best_k = k
        best_s = s + 1  # 1-based index
        
    # Random search
    random.seed(42)
    
    while time.time() < deadline:
        L = random.randint(1, max_n - W)
        phi_arr = compute_phi_segment(L, W)
        s, k = find_longest_distinct(phi_arr)
        if k > best_k:
            best_k = k
            best_s = L + s
            
    return {'start': best_s, 'length': best_k}
