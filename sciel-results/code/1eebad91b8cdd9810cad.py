import time
import math
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    def sieve_primes(limit):
        """Generate list of primes up to limit using sieve of Eratosthenes."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 10^6 for fast factorization of individual numbers
    primes = sieve_primes(1000000)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes."""
        if x <= 1:
            return x if x == 1 else 0
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def segmented_sieve_spf(L, R, prime_limit):
        """
        Segmented sieve to compute smallest prime factor for each number in [L, R].
        Returns a list spf where spf[i] is the smallest prime factor of (L+i), or 0 if prime.
        Actually, let's compute phi directly in a segmented way.
        """
        n = R - L + 1
        phi = list(range(L, R + 1))
        
        # For each prime p, mark multiples in [L, R]
        for p in primes:
            if p > prime_limit:
                break
            # Find first multiple of p in [L, R]
            if p * p > R:
                continue
            
            first = ((L + p - 1) // p) * p
            if first < p:
                first = p
            
            for j in range(first, R + 1, p):
                idx = j - L
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        # For remaining primes where p*p > R, we need to handle the case where
        # the number itself is prime or has a large prime factor
        # Actually, the above loop handles all primes p where p*p <= R.
        # For numbers that still have a factor > sqrt(original), we need to handle them.
        # But wait, the segmented sieve approach above is incomplete.
        # Let me use a different approach: compute phi for each number individually
        # but use the precomputed primes for efficiency.
        
        # Actually, let's just use phi_fast for each number, but batch them.
        # For a window of size 10^6, that's 10^6 calls to phi_fast, which should be fast enough.
        
        return phi
    
    def compute_phi_segment(L, R):
        """Compute phi for all n in [L, R]."""
        result = []
        for n in range(L, R + 1):
            result.append(phi_fast(n))
        return result
    
    def find_longest_distinct_run(phi_vals, start):
        """
        Given a list of phi values for consecutive integers starting at 'start',
        find the longest contiguous subarray where all values are distinct.
        Returns (length, index_of_start_in_phi_vals).
        """
        n = len(phi_vals)
        if n == 0:
            return 0, 0
        
        best_len = 1
        best_idx = 0
        seen = set()
        left = 0
        
        for right in range(n):
            while phi_vals[right] in seen and left <= right:
                seen.remove(phi_vals[left])
                left += 1
            seen.add(phi_vals[right])
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_idx = left
        
        return best_len, best_idx
    
    best_k = 1
    best_s = 1
    
    def update_best(s, k):
        nonlocal best_k, best_s
        if k > best_k:
            best_k = k
            best_s = s
    
    # Strategy 1: Scan a large contiguous range near 1
    # Start with 1 to 1,000,000
    if time.time() < deadline - 30:
        L, R = 1, 1000000
        phi_vals = compute_phi_segment(L, R)
        k, idx = find_longest_distinct_run(phi_vals, L)
        s = L + idx
        update_best(s, k)
    
    # Strategy 2: Scan multiple windows at different scales
    # Focus on ranges where collisions are less likely
    windows_to_try = [
        (10**4, 10**4 + 10**6),
        (10**5, 10**5 + 10**6),
        (10**6, 10**6 + 10**6),
        (10**7, 10**7 + 10**6),
        (10**8, 10**8 + 10**6),
    ]
    
    for L, R in windows_to_try:
        if time.time() > deadline - 30:
            break
        if R > 10**12:
            R = 10**12
        
        # If window is too large, shrink it
        if R - L > 500000:
            R = L + 500000
        
        phi_vals = compute_phi_segment(L, R)
        k, idx = find_longest_distinct_run(phi_vals, L)
        s = L + idx
        if s + k - 1 <= 10**12:
            update_best(s, k)
    
    # Strategy 3: Random sampling with local refinement
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10 and attempts < 500:
        attempts += 1
        
        # Sample from various ranges
        r = random.random()
        if r < 0.2:
            s = random.randint(10**4, 10**7)
        elif r < 0.4:
            s = random.randint(10**7, 10**9)
        elif r < 0.6:
            s = random.randint(10**9, 10**11)
        elif r < 0.8:
            s = random.randint(10**11, 10**12 - 1000)
        else:
            s = random.randint(10**5, 10**12 - 1000)
        
        if s < 1 or s + 1000 > 10**12:
            continue
        
        # Check a window of size 1000 around s
        L = max(1, s - 500)
        R = min(10**12, s + 500)
        
        if R - L < 100:
            continue
        
        phi_vals = compute_phi_segment(L, R)
        k, idx = find_longest_distinct_run(phi_vals, L)
        s_found = L + idx
        if s_found + k - 1 <= 10**12:
            update_best(s_found, k)
    
    # Strategy 4: If we have time, try to extend the best run found
    if time.time() < deadline - 5 and best_k > 1:
        # Try to extend left and right from the best run
        for _ in range(3):
            s_try = best_s - 100
            if s_try < 1:
                s_try = 1
            L = max(1, best_s - 500)
            R = min(10**12, best_s + 500)
            phi_vals = compute_phi_segment(L, R)
            k, idx = find_longest_distinct_run(phi_vals, L)
            s_found = L + idx
            if s_found + k - 1 <= 10**12:
                update_best(s_found, k)
            
            s_try = best_s + 100
            L = max(1, best_s - 500)
            R = min(10**12, best_s + 500)
            phi_vals = compute_phi_segment(L, R)
            k, idx = find_longest_distinct_run(phi_vals, L)
            s_found = L + idx
            if s_found + k - 1 <= 10**12:
                update_best(s_found, k)
    
    # Ensure validity
    s = best_s
    k = best_k
    if s < 1:
        s = 1
    if s + k - 1 > 10**12:
        k = 10**12 - s + 1
        if k < 1:
            s = 1
            k = 1
    if k > 200:
        k = 200
    if k < 1:
        k = 1
    
    return {'start': s, 'length': k}
