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
    
    max_n = start + length - 1
    sqrt_max = int(math.isqrt(max_n)) + 1
    
    for p in primes:
        if p > sqrt_max:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > max_n:
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

def check_distinct(phis):
    """Check if all values in phis are distinct. Return True/False."""
    return len(set(phis)) == len(phis)

def find_longest_distinct_prefix(phis):
    """Find the longest prefix of phis with all distinct values."""
    seen = set()
    for i, p in enumerate(phis):
        if p in seen:
            return i
        seen.add(p)
    return len(phis)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    # STRATEGY 1: Deterministic candidates near 10^12
    # For large s, collision probability is ~k^2/s which is tiny.
    # Try several deterministic offsets from 10^12 - 200.
    base = max_n - max_k + 1  # = 999999999801, so base + 199 = 10^12
    
    deterministic_starts = []
    # Try offsets 0, 1, 2, ... and some spaced ones
    for off in [0, 1, 2, 3, 5, 10, 17, 23, 31, 50, 73, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000]:
        s = base - off
        if s >= 1:
            deterministic_starts.append(s)
    
    # Also try some mid-range deterministic points
    deterministic_starts.extend([
        999999999999 - max_k + 1,
        999999999000 - max_k + 1,
        999999000000 - max_k + 1,
        999000000000 - max_k + 1,
        900000000000 - max_k + 1,
        500000000000 - max_k + 1,
    ])
    
    # Deduplicate while preserving order
    seen_starts = set()
    unique_starts = []
    for s in deterministic_starts:
        if s not in seen_starts and s >= 1 and s + max_k - 1 <= max_n:
            seen_starts.add(s)
            unique_starts.append(s)
    
    for s in unique_starts:
        if time.time() > deadline - 30:
            break
        phis = compute_phi_segment(s, max_k, primes)
        if check_distinct(phis):
            return {'start': s, 'length': max_k}
        run_len = find_longest_distinct_prefix(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s
    
    # STRATEGY 2: Random search in high range
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline - 20:
            break
        s = random.randint(max(1, max_n - 10**8), max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if check_distinct(phis):
            return {'start': s, 'length': max_k}
        run_len = find_longest_distinct_prefix(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s
    
    # STRATEGY 3: Random search across all ranges
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if check_distinct(phis):
            return {'start': s, 'length': max_k}
        run_len = find_longest_distinct_prefix(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s
    
    # STRATEGY 4: Local refinement around best found
    if best_k >= 50 and time.time() < deadline - 5:
        for delta in range(-200, 201, 1):
            if time.time() > deadline - 3:
                break
            s = best_start + delta
            if s < 1 or s + max_k - 1 > max_n:
                continue
            phis = compute_phi_segment(s, max_k, primes)
            if check_distinct(phis):
                return {'start': s, 'length': max_k}
            run_len = find_longest_distinct_prefix(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s
    
    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
