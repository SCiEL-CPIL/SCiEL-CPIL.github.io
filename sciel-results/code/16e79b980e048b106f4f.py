"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    for p in primes:
        if p * p > start + length - 1:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > start + length - 1:
            continue
        
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest contiguous subarray with all distinct values.
    Returns (length, start_index)."""
    n = len(phis)
    if n == 0:
        return 0, 0
    best = 1
    best_start = 0
    left = 0
    seen = {}
    for right, val in enumerate(phis):
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        current_len = right - left + 1
        if current_len > best:
            best = current_len
            best_start = left
    return best, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    max_k = 200
    max_n = 10**12
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_k = 1
    best_start = 1
    
    # Phase 1: Large segmented sieve scan
    # Scan large windows to find long runs of distinct phi values
    window_size = 500000  # 500K numbers per window
    num_windows = 20  # Try 20 different windows
    
    random.seed(42)
    
    # Choose window start positions strategically
    # Focus on ranges where distinct phi values are more likely
    # Larger numbers tend to have more distinct phi values in a given range
    
    window_starts = []
    # Small range for quick check
    window_starts.append(1)
    # Medium ranges
    for mag in [4, 5, 6, 7, 8, 9, 10, 11]:
        lo = 10**(mag - 1)
        hi = min(10**mag - window_size, max_n - window_size)
        if hi > lo:
            # Pick a few random positions in this range
            for _ in range(2):
                s = random.randint(lo, hi)
                window_starts.append(s)
    
    # Remove duplicates and sort
    window_starts = list(set(window_starts))
    window_starts.sort()
    
    for ws in window_starts:
        if time.time() > deadline - 30:
            break
        if ws < 1 or ws + window_size > max_n:
            continue
        
        # Compute phi for this window
        phis = compute_phi_segment(ws, window_size, primes)
        
        # Find longest distinct run in this window
        run_len, run_start = find_longest_distinct_run(phis)
        
        if run_len > best_k:
            best_k = run_len
            best_start = ws + run_start
            
            # If we found a very long run, check if it's maximal
            if best_k >= 150:
                # Try to extend this run
                extend_left = min(1000, ws)
                extend_right = min(1000, max_n - (ws + window_size))
                
                # Check left extension
                for delta in range(1, extend_left + 1):
                    if time.time() > deadline - 25:
                        break
                    s = best_start - delta
                    if s < 1:
                        break
                    # Check if phi(s) is distinct from phi(s+1) to phi(best_start + best_k - 2)
                    phi_s = compute_phi_segment(s, 1, primes)[0]
                    # Check against the existing run
                    existing_phis = phis[best_start - ws : best_start - ws + best_k - 1]
                    if phi_s not in existing_phis:
                        best_k += 1
                        best_start = s
                    else:
                        break
                
                # Check right extension
                for delta in range(1, extend_right + 1):
                    if time.time() > deadline - 25:
                        break
                    s = best_start + best_k - 1 + delta
                    if s >= max_n:
                        break
                    phi_s = compute_phi_segment(s, 1, primes)[0]
                    existing_phis = phis[best_start - ws : best_start - ws + best_k - 1]
                    if phi_s not in existing_phis:
                        best_k += 1
                    else:
                        break
    
    # Phase 2: Local refinement around best found run
    if best_k >= 10 and time.time() < deadline - 20:
        search_radius = min(5000, max(500, best_k * 20))
        for delta in range(-search_radius, search_radius + 1, 10):
            if time.time() > deadline - 15:
                break
            s = best_start + delta
            if s < 1 or s + max_k - 1 > max_n:
                continue
            phis = compute_phi_segment(s, max(200, best_k + 50), primes)
            run_len, run_start = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_start
    
    # Phase 3: Random sampling in unexplored regions
    if time.time() < deadline - 10:
        num_trials = 50
        for i in range(num_trials):
            if time.time() > deadline - 5:
                break
            magnitude = random.choice([5, 6, 7, 8, 9, 10, 11])
            lo = 10**(magnitude - 1)
            hi = min(10**magnitude - max_k, max_n - max_k)
            if hi <= lo:
                continue
            s = random.randint(lo, hi)
            phis = compute_phi_segment(s, max_k, primes)
            if len(set(phis)) == max_k:
                return {'start': s, 'length': max_k}
            run_len, run_start = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_start
    
    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
