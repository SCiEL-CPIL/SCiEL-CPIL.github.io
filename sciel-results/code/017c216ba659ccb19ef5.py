"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
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
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        primes = [i for i in range(2, limit + 1) if is_prime[i]]
        return primes
    
    def compute_phi_numpy(lo, hi):
        """Compute phi for numbers in [lo, hi] using NumPy-accelerated segmented sieve."""
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)
        
        # Initialize phi and remaining arrays
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            # Find indices where the number is divisible by p
            start_idx = ((lo + p - 1) // p) * p - lo
            if start_idx < 0:
                start_idx = 0
            
            # Create mask of indices divisible by p
            indices = np.arange(start_idx, n, p)
            
            # Check which of these still have p as a factor in remaining
            mask = (remaining[indices] % p == 0)
            valid_indices = indices[mask]
            
            if len(valid_indices) > 0:
                # Update phi: phi[n] -= phi[n] // p
                phi[valid_indices] -= phi[valid_indices] // p
                
                # Remove all factors of p from remaining
                for idx in valid_indices:
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > limit
        mask = remaining > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // remaining[mask]
        
        return phi
    
    def find_best_run_numpy(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values using a rolling approach."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        
        best_k = 0
        best_s = offset
        
        # Use a dictionary to track last seen position for each phi value
        last_seen = {}
        
        # Sliding window approach
        left = 0
        for right in range(n):
            val = phi_arr[right]
            
            # If we've seen this value before and it's within our window, move left
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            
            last_seen[val] = right
            
            # Current window length
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = offset + left
            
            # Cap at max_len
            if best_k >= max_len:
                break
        
        return best_s, min(best_k, max_len)
    
    best_s, best_k = 1, 1
    
    # Strategy: Search larger contiguous blocks where we can fit many numbers
    # and find the longest run of distinct phi values.
    # Key insight: For large n, phi(n) values tend to be spread out,
    # making long runs of distinct values more likely.
    
    # Test ranges - progressively larger blocks
    # We want blocks of at least 500-1000 numbers to find runs up to 200
    test_ranges = [
        (1, 5000),
        (10000, 15000),
        (50000, 55000),
        (100000, 105000),
        (500000, 505000),
        (1000000, 1005000),
        (5000000, 5005000),
        (10000000, 10005000),
        (50000000, 50005000),
        (100000000, 100005000),
    ]
    
    for lo, hi in test_ranges:
        if time_left() < 8.0:
            break
        try:
            phi_arr = compute_phi_numpy(lo, hi)
            if time_left() < 5.0:
                break
            s, k = find_best_run_numpy(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # If we found a good run, try to extend it by checking nearby ranges
    if best_k > 50 and time_left() > 25.0:
        # Try a few offsets around the best found run
        for delta in [-500, -200, 200, 500]:
            if time_left() < 10.0:
                break
            new_lo = max(1, best_s + delta)
            new_hi = new_lo + 2000
            if new_hi > 10**12:
                continue
            try:
                phi_arr = compute_phi_numpy(new_lo, new_hi)
                if time_left() < 5.0:
                    break
                s, k = find_best_run_numpy(phi_arr, new_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Try some very large ranges if we have time
    if time_left() > 40.0:
        more_ranges = [
            (1000000000, 1000010000),
            (10000000000, 10000010000),
            (100000000000, 100000010000),
        ]
        for lo, hi in more_ranges:
            if time_left() < 15.0:
                break
            try:
                phi_arr = compute_phi_numpy(lo, hi)
                if time_left() < 10.0:
                    break
                s, k = find_best_run_numpy(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
