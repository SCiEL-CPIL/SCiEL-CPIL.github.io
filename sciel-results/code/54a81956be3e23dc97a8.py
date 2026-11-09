import math
import random
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 40s buffer for safety
    
    # 1. Generate powerful numbers up to L
    # Powerful numbers: n = a^2 * b^3 where b is square-free
    # For efficiency, we generate up to a reasonable limit
    # L = 5*10^8 gives ~300K powerful numbers, manageable for O(P^2) with NumPy
    # L = 2*10^9 gives ~1M powerful numbers
    L = 2 * 10**9
    
    powerful_set = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = math.isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
        
        if time.time() > deadline - 60:
            break
            
    # Convert to sorted list
    powerful_list = sorted(powerful_set)
    N_pow = len(powerful_list)
    
    if N_pow < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # 2. Use NumPy to compute all pairwise sums and count them
    # We want to find N with many representations as sum of two powerful numbers
    # Strategy: sample a subset of powerful numbers, compute all pairwise sums,
    # find the most common sum, then verify all pairs for that sum
    
    # Take a subset to keep O(P^2) manageable
    # With NumPy, we can handle ~50K x 50K = 2.5B operations... too much
    # Better: take top portion and use smarter sampling
    # Actually, let's take all powerful numbers but be smart about it
    
    # Convert to numpy array
    pow_arr = np.array(powerful_list, dtype=np.int64)
    
    # For O(P^2) counting, we need P to be manageable
    # If P > 100000, we'll sample
    if N_pow > 100000:
        # Sample from the larger values (more likely to produce interesting sums)
        # Take top 100K powerful numbers
        sample_start = max(0, N_pow - 100000)
        pow_arr = pow_arr[sample_start:]
        N_pow = len(pow_arr)
    
    # Compute all pairwise sums (i <= j to avoid double counting)
    # Use broadcasting: pow_arr[:, None] + pow_arr[None, :]
    # This creates a P x P matrix. For P=100K, that's 10^10 elements = 80GB. Too much!
    # We need a different approach.
    
    # Alternative: use the fact that we only care about sums up to 10^18
    # And we want to find the most frequent sum.
    # 
    # Better approach: for each powerful number a, compute N - a for candidate N values
    # and check if N - a is also powerful.
    #
    # Even better: use a hash-based approach with random sampling of N values
    # and count how many pairs (a, b) with a + b = N exist.
    
    # Let's use a hybrid: 
    # 1. Generate candidate N values from sums of random pairs
    # 2. For each candidate N, count all pairs (a, b) with a + b = N
    
    # Step 1: Sample many random pairs to get candidate N values
    n_samples = 50000
    if N_pow < n_samples:
        n_samples = max(100, N_pow // 2)
    
    # Random indices
    idx_a = np.random.randint(0, N_pow, size=n_samples)
    idx_b = np.random.randint(0, N_pow, size=n_samples)
    
    a_vals = pow_arr[idx_a]
    b_vals = pow_arr[idx_b]
    
    # Ensure a <= b
    mask = a_vals <= b_vals
    a_vals = a_vals[mask]
    b_vals = b_vals[mask]
    
    sums = a_vals + b_vals
    sums = sums[sums <= 10**18]
    
    if len(sums) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count frequency of each sum
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    # Get top 10 candidate N values
    top_k = min(10, len(unique_sums))
    top_indices = np.argsort(counts)[::-1][:top_k]
    top_candidates = unique_sums[top_indices]
    
    best_N = 0
    best_pairs = []
    best_score = 0
    
    # Step 2: For each candidate N, find ALL pairs (a, b) with a + b = N
    # where both a and b are powerful and a <= b
    for N in top_candidates:
        if time.time() > deadline - 15:
            break
            
        N = int(N)
        pairs = []
        
        # Iterate through powerful numbers a <= N//2
        # Check if N - a is in powerful_set
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
                
        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N
            best_pairs = pairs
    
    if best_N == 0:
        # Fallback: try N = 2
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
