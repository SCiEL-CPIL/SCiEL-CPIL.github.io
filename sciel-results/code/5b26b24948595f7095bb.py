"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with sieve-based square-free check."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status using sieve-like approach
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if sq_free[p]:
            p2 = p * p
            for m in range(p2, max_b + 1, p2):
                sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if b > max_b:
            break
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def _count_pairs_for_N(N, powerful, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, both powerful."""
    count = 0
    for a in powerful:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set and b >= a:
            count += 1
    return count

def _generate_pairs_for_N(N, powerful, powerful_set):
    """Generate all pairs (a, b) with a <= b, a + b = N, both powerful."""
    pairs = []
    for a in powerful:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    return pairs

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    # Use a moderately large limit to balance generation time and search space
    limit = 60_000_000  # 6 * 10^7
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Use FFT convolution to count all sums efficiently
    # Create indicator vector: f[i] = 1 if i is powerful, else 0
    # Then (f * f)[s] = number of ordered pairs (a,b) with a+b=s where both are powerful
    
    max_sum = 2 * limit
    # Use float32 for memory efficiency
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    # Create indicator array
    f = np.zeros(n_fft, dtype=np.float32)
    for val in powerful:
        if val < n_fft:
            f[val] = 1.0
    
    # FFT convolution
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=n_fft)
    
    # Round to integers
    conv_int = np.round(conv[:max_sum + 1]).astype(np.int64)
    
    # Compute number of pairs with a <= b for each sum s
    # For even s where s/2 is powerful, add 1 before dividing by 2
    pairs_count = np.zeros(max_sum + 1, dtype=np.int64)
    
    even_s = np.arange(0, max_sum + 1, 2)
    half_s = even_s // 2
    is_powerful_half = (f[half_s] == 1.0)
    
    adjusted = conv_int.copy()
    adjusted[even_s[is_powerful_half]] += 1
    
    pairs_count = adjusted // 2
    
    # Exclude N=0 and N=1
    if max_sum >= 2:
        pairs_count[:2] = 0
    
    # Find top candidates
    # Get indices sorted by count (descending)
    top_k = 50
    if len(pairs_count) > top_k:
        top_indices = np.argsort(pairs_count)[-top_k:][::-1]
    else:
        top_indices = np.arange(len(pairs_count))
    
    # Verify top candidates and find the best
    best_N = 2
    best_count = 0
    best_pairs = [[1, 1]]
    
    for idx in top_indices:
        N = int(idx)
        if N < 2:
            continue
        
        # Time check
        if time.time() > deadline - 20:
            break
        
        # Count pairs for this N
        count = _count_pairs_for_N(N, powerful, powerful_set)
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = _generate_pairs_for_N(N, powerful, powerful_set)
    
    # If we have time, try with a larger limit
    remaining = deadline - time.time()
    if remaining > 50 and best_count < 15:
        limit2 = 100_000_000
        powerful2 = _generate_powerful(limit2)
        powerful_set2 = set(powerful2)
        K2 = len(powerful2)
        
        if K2 > K:
            arr2 = np.array(powerful2, dtype=np.int64)
            max_sum2 = 2 * limit2
            n_fft2 = 1
            while n_fft2 < max_sum2 + 1:
                n_fft2 *= 2
            
            f2 = np.zeros(n_fft2, dtype=np.float32)
            for val in powerful2:
                if val < n_fft2:
                    f2[val] = 1.0
            
            F2 = np.fft.rfft(f2)
            conv2 = np.fft.irfft(F2 * F2, n=n_fft2)
            conv2_int = np.round(conv2[:max_sum2 + 1]).astype(np.int64)
            
            pairs_count2 = np.zeros(max_sum2 + 1, dtype=np.int64)
            even_s2 = np.arange(0, max_sum2 + 1, 2)
            half_s2 = even_s2 // 2
            is_powerful_half2 = (f2[half_s2] == 1.0)
            
            adjusted2 = conv2_int.copy()
            adjusted2[even_s2[is_powerful_half2]] += 1
            pairs_count2 = adjusted2 // 2
            if max_sum2 >= 2:
                pairs_count2[:2] = 0
            
            top_indices2 = np.argsort(pairs_count2)[-top_k:][::-1]
            
            for idx in top_indices2:
                N = int(idx)
                if N < 2:
                    continue
                
                if time.time() > deadline - 10:
                    break
                
                count = _count_pairs_for_N(N, powerful2, powerful_set2)
                
                if count > best_count:
                    best_count = count
                    best_N = N
                    best_pairs = _generate_pairs_for_N(N, powerful2, powerful_set2)
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
