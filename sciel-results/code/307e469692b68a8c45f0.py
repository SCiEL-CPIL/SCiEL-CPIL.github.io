"""Improved program: high-base search with collision-avoidance heuristic."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented sieve with NumPy."""
        n = hi - lo + 1
        if n <= 0:
            return []
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)
        
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            idxs = np.arange(start - lo, n, p)
            if len(idxs) == 0:
                continue
            
            phi[idxs] = phi[idxs] // p * (p - 1)
            
            rem_sel = remaining[idxs].copy()
            for _ in range(50):
                mask = rem_sel % p == 0
                if not mask.any():
                    break
                rem_sel[mask] = rem_sel[mask] // p
            remaining[idxs] = rem_sel
        
        mask = remaining > 1
        if np.any(mask):
            phi[mask] = phi[mask] // remaining[mask] * (remaining[mask] - 1)
        
        return phi.tolist()
    
    def find_best_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        best_k = 0
        best_s = offset
        left = 0
        seen = {}
        
        for right in range(n):
            val = phi_arr[right]
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = min(current_len, max_len)
                best_s = offset + left
            if best_k >= max_len:
                break
        
        return best_s, best_k
    
    def collision_score(s, k, primes):
        """
        Probabilistic collision-avoidance heuristic:
        Score how likely phi(s+i) values are to collide.
        Lower score = better (fewer expected collisions).
        
        For each pair (i,j) with i<j<k, phi(s+i)=phi(s+j) is more likely
        if s+i and s+j share large prime factors or have similar structure.
        We approximate by checking divisibility patterns.
        """
        score = 0
        # Check for small-prime divisibility patterns that increase collision risk
        for p in primes[:30]:  # First 30 primes
            count_div = 0
            for i in range(k):
                if (s + i) % p == 0:
                    count_div += 1
            # If many numbers are divisible by p, phi values tend to be "smaller"
            # and more likely to collide
            if count_div > k * 0.1:
                score += count_div
        return score
    
    best_s, best_k = 1, 1
    
    random.seed(42)
    
    max_n = 10**12
    
    # Pre-sieve primes up to 10^6 once
    primes_1e6 = sieve_primes(10**6)
    
    # Strategy 1: Search in high range where collisions are rare
    # At n ~ 10^11+, phi values are large and distinct with high probability
    SEGMENT_SIZE = 100000
    max_start = max_n - SEGMENT_SIZE
    
    # Try several random segments in the high range
    # Focus on [10^10, 10^12] where collisions are extremely rare
    high_lo = 10**10
    high_hi = max_n - SEGMENT_SIZE
    
    attempts = 0
    while time_left() > 20.0 and attempts < 50:
        attempts += 1
        # Random start in high range
        s = random.randint(high_lo, high_hi)
        hi = s + SEGMENT_SIZE - 1
        
        try:
            phi_arr = compute_phi_segmented(s, hi)
            run_s, run_k = find_best_run(phi_arr, s, max_len=200)
            if run_k > best_k:
                best_k = run_k
                best_s = run_s
                if best_k >= 200:
                    break
        except Exception:
            pass
    
    # If we found k=200, we're done
    if best_k >= 200:
        # Verify and return
        if best_s + best_k - 1 > max_n:
            best_k = max_n - best_s + 1
        if best_k < 1:
            best_k = 1
            best_s = 1
        return {'start': best_s, 'length': best_k}
    
    # Strategy 2: If we haven't found 200, try more targeted search
    # Use collision-avoidance heuristic to pick better starting points
    while time_left() > 10.0:
        # Try different ranges
        range_lo = random.randint(10**9, max_n - SEGMENT_SIZE)
        s = range_lo
        hi = s + SEGMENT_SIZE - 1
        
        try:
            phi_arr = compute_phi_segmented(s, hi)
            run_s, run_k = find_best_run(phi_arr, s, max_len=200)
            if run_k > best_k:
                best_k = run_k
                best_s = run_s
                if best_k >= 200:
                    break
        except Exception:
            pass
    
    # Refinement: expand around best found
    if best_k >= 2 and time_left() > 15.0:
        for delta in [-2000, -1000, 1000, 2000, 5000]:
            if time_left() < 5.0:
                break
            extend_lo = max(1, best_s + delta - 500)
            extend_hi = min(max_n, best_s + best_k + 500 + delta)
            if extend_hi - extend_lo < 100:
                continue
            try:
                phi_arr = compute_phi_segmented(extend_lo, extend_hi)
                s, k = find_best_run(phi_arr, extend_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > max_n:
        best_k = min(best_k, max_n - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
