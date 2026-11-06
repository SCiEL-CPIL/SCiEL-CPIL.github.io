import time
from math import isqrt
from collections import Counter
import numpy as np

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > temp:
            break
        if temp % (p * p) == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    max_b = int(limit ** (1/3)) + 2
    
    # Pre-compute small primes for square-free check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
            while t % p == 0:
                t //= p
        if not is_sf:
            continue
        if t > 1:
            r = isqrt(t)
            if r * r == t:
                is_sf = False
        if not is_sf:
            continue
            
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Generate powerful numbers up to a limit
    # L = 3 * 10^8
    pow_limit = 3 * 10**8
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 50:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}

    # Count sums using NumPy chunking
    counts = Counter()
    P = np.array(powerful, dtype=np.int64)
    
    # Process in chunks
    chunk_size = 500
    n_chunks = (K + chunk_size - 1) // chunk_size
    
    for c in range(n_chunks):
        start_idx = c * chunk_size
        end_idx = min(start_idx + chunk_size, K)
        P_chunk = P[start_idx:end_idx]
        
        # Compute sums P_chunk[i] + P[j] for all j
        Sums = P_chunk[:, None] + P[None, :]
        # Keep only pairs where P[j] >= P_chunk[i] to ensure a <= b and avoid duplicates
        Mask = P[None, :] >= P_chunk[:, None]
        
        valid_sums = Sums[Mask]
        
        if len(valid_sums) > 0:
            # Count occurrences in this chunk
            unique_vals, unique_counts = np.unique(valid_sums, return_counts=True)
            for val, cnt in zip(unique_vals, unique_counts):
                counts[val] += cnt
                
        if time.time() > deadline - 20:
            break

    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    best_N = max(counts, key=counts.get)
    
    # Extract pairs
    pow_set = set(powerful)
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
            
    # Verify
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
            
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
        
    return {'N': best_N, 'pairs': verified_pairs}
