"""Sieve-based phi computation with sliding window for long distinct-totient runs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    MAX_N = 10**12
    MAX_K = 200
    WINDOW = 2_000_000  # size of segmented sieve window

    # Generate primes up to sqrt(MAX_N) using simple sieve
    limit = int(math.isqrt(MAX_N)) + 1
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]

    best_k = 1
    best_s = 1

    # We'll scan windows across [1, MAX_N]
    # For each window, compute phi for all numbers, then find longest run of distinct phi values
    
    start = 1
    while start <= MAX_N:
        if time.time() > deadline - 5:
            break
        
        end = min(start + WINDOW - 1, MAX_N)
        wsize = end - start + 1
        
        # Initialize phi array for this window
        # phi[i] starts as i (the number itself)
        nums = np.arange(start, end + 1, dtype=np.int64)
        phi_vals = nums.copy()
        
        # For each prime p, find multiples in [start, end] and update phi
        for p in primes:
            if p * p > end:
                break
            # First multiple of p >= start
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            # Update phi for all multiples of p in the window
            # Indices: first - start, first - start + p, ...
            idx = first - start
            while idx < wsize:
                # phi[n] = phi[n] / p * (p-1)
                phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                idx += p
        
        # Handle remaining prime factors > sqrt(end)
        # For numbers that still have a factor > 1 after removing all prime factors <= sqrt(end)
        # That remaining factor is prime
        for i in range(wsize):
            if phi_vals[i] != 0:
                # Check if nums[i] has been fully reduced
                # Actually, we need a different approach: track remaining factor
                pass
        
        # Better approach: use a segmented sieve that tracks smallest prime factor or remaining value
        # Let's redo this more carefully
        
        # Re-initialize
        phi_vals = nums.copy()
        remaining = nums.copy()  # track remaining unfactored part
        
        for p in primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            idx = first - start
            while idx < wsize:
                phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                # Remove all factors of p from remaining
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
                idx += p
        
        # For any remaining factor > 1, it's a prime factor
        for i in range(wsize):
            r = remaining[i]
            if r > 1:
                phi_vals[i] = phi_vals[i] // r * (r - 1)
        
        # Now find longest run of consecutive distinct phi values
        # Use a sliding window approach
        seen = set()
        left = 0
        for right in range(wsize):
            seen.add(phi_vals[right])
            while phi_vals[right] in seen and len(seen) > 1:
                # This check is wrong; let me use a different approach
                break
        
        # Simpler: for each starting position, extend as far as possible
        # But that's O(W^2) which is too slow for W=2M
        # Use two-pointer / sliding window with frequency count
        
        freq = {}
        left = 0
        max_run_in_window = 0
        for right in range(wsize):
            val = phi_vals[right]
            freq[val] = freq.get(val, 0) + 1
            while freq[val] > 1:
                left_val = phi_vals[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            run_len = right - left + 1
            if run_len > max_run_in_window:
                max_run_in_window = run_len
                if run_len > best_k:
                    best_k = run_len
                    best_s = start + left
        
        start = end + 1

    # Verify the best result with exact phi computation
    def phi_exact(x):
        result, d, y = x, 2, x
        while d * d <= y:
            if y % d == 0:
                result -= result // d
                while y % d == 0:
                    y //= d
            d += 1
        return result - result // y if y > 1 else result

    # Verify and possibly adjust
    verified_k = 1
    verified_s = best_s
    for s in range(max(1, best_s - 5), best_s + 5):
        if s + 200 - 1 > MAX_N:
            break
        seen = set()
        k = 0
        for i in range(200):
            n = s + i
            if n > MAX_N:
                break
            p = phi_exact(n)
            if p in seen:
                break
            seen.add(p)
            k += 1
        if k > verified_k:
            verified_k = k
            verified_s = s

    return {'start': verified_s, 'length': verified_k}
# EVOLVE-BLOCK-END
