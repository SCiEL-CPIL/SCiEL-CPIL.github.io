import time
import math
import numpy as np

def get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def compute_phi_segment(L, M, primes):
    """
    Compute phi(n) for n in [L, L+M-1].
    """
    # n is the array of numbers
    n = L + np.arange(M, dtype=np.int64)
    phi = n.copy()
    rem = n.copy()
    
    R = L + M - 1
    sqrt_R = int(math.isqrt(R))
    
    for p in primes:
        if p > sqrt_R:
            break
        
        # Find first multiple of p in [L, R]
        start = (L + p - 1) // p * p
        if start < L:
            start += p
            
        # Indices in the segment
        # indices are relative to L
        indices = np.arange(start - L, M, p, dtype=np.int64)
        
        if len(indices) == 0:
            continue
            
        # Update phi
        phi[indices] -= phi[indices] // p
        
        # Update rem: divide by p as much as possible
        # Max power of p in any n <= 10^12 is floor(log_p(10^12))
        max_power = int(math.log(10**12, p)) + 1
        for _ in range(max_power):
            vals = rem[indices]
            mask = vals % p == 0
            if not mask.any():
                break
            target_indices = indices[mask]
            rem[target_indices] //= p
            
    # Handle remaining prime factors > sqrt(R)
    # rem[i] will be 1 if n had no large prime factor, or the large prime factor itself
    mask = rem > 1
    if mask.any():
        # For these indices, rem[i] is the prime factor q
        # phi[i] currently is phi(a) * q, where n = a * q
        # We need to subtract phi[i] // q
        # So phi[i] -= phi[i] // rem[i]
        phi[mask] -= phi[mask] // rem[mask]
        
    return phi

def find_longest_distinct_run(phi_vals):
    """
    Find the longest contiguous subsequence with distinct values.
    Returns (max_length, start_index_in_phi_vals)
    """
    n = len(phi_vals)
    if n < 2:
        return 1, 0
        
    last_seen = {}
    left = 0
    max_len = 0
    max_start = 0
    
    for right in range(n):
        val = phi_vals[right]
        if val in last_seen and last_seen[val] >= left:
            left = last_seen[val] + 1
        last_seen[val] = right
        
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
            max_start = left
            
    return max_len, max_start

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin (150s kill, 15s buffer)
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
        
    # Precompute primes up to 10^6
    # sqrt(10^12) = 10^6
    primes = get_primes(10**6)
    
    best_k = 1
    best_s = 1
    
    # Search strategy:
    # Try segments near 10^12, as phi values are larger and collisions are less likely.
    # Use smaller segments to allow more search, or larger for fewer but deeper searches.
    # Let's use M = 500000 to balance speed and coverage.
    M = 500000
    
    base = 10**12
    max_L = base - M + 1
    
    # Generate many candidate segments
    segments = []
    
    # Dense coverage near the top (most promising region)
    num_top = 20
    for i in range(num_top):
        L = max_L - i * M
        if L >= 1:
            segments.append(L)
    
    # Scattered segments across the range
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        L = int(base * frac) - M
        if L < 1:
            L = 1
        if L + M - 1 > base:
            L = base - M + 1
        segments.append(L)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_segments = []
    for L in segments:
        if L not in seen:
            seen.add(L)
            unique_segments.append(L)
    segments = unique_segments
    
    for L in segments:
        if time_remaining() < 15.0:
            break
            
        try:
            t0 = time.time()
            phi_vals = compute_phi_segment(L, M, primes)
            compute_time = time.time() - t0
            
            # phi_vals[0] corresponds to n = L
            k, rel_start = find_longest_distinct_run(phi_vals)
            s = L + rel_start
            
            # Check constraints
            if s < 1:
                continue
            if s + k - 1 > base:
                k = base - s + 1
                if k < 1:
                    continue
                    
            if k > 200:
                k = 200
                
            if k > best_k:
                best_k = k
                best_s = s
                
                if best_k == 200:
                    # Max possible length found
                    break
                    
        except Exception as e:
            # If something goes wrong, skip this segment
            continue
            
    # If we didn't find anything good, default to 1
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    return {'start': best_s, 'length': best_k}
