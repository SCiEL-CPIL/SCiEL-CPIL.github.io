"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is square-free.
    However, generating for all b (not just square-free) and taking a set is simpler and correct,
    as duplicates are handled by the set.
    """
    if limit < 1:
        return []
    
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerfuls.add(n)
    
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s buffer for 150s limit
    
    # Phase 1: Generate powerful numbers up to a large limit
    # The count of powerful numbers up to L is ~ c * sqrt(L)
    # For L=10^8, count ~ 3000-5000
    # For L=10^10, count ~ 30000-50000
    # For L=10^12, count ~ 300000-500000
    
    # We'll do a two-phase approach:
    # Phase 1: Use np to count pairwise sums for a moderate limit to find top candidates
    # Phase 2: Verify top candidates against a larger set
    
    # Phase 1: Find candidates using NumPy for speed
    L1 = 10**8
    pow_list_1 = np.array(generate_powerful_numbers(L1), dtype=np.int64)
    n1 = len(pow_list_1)
    
    # Use broadcasting to compute all pairwise sums (i <= j)
    # This creates an n1 x n1 array, so we need n1 to be manageable
    # For n1 ~ 5000, that's 25M entries * 8 bytes = 200MB, which is okay
    # But let's be careful and use a chunked approach if needed
    
    # Actually, let's use a more memory-efficient approach:
    # Compute upper triangle sums
    i_idx, j_idx = np.triu_indices(n1)
    sums = pow_list_1[i_idx] + pow_list_1[j_idx]
    
    # Count occurrences
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    # Get top 100 candidates
    top_idx = np.argsort(-counts)[:100]
    candidates = unique_sums[top_idx].tolist()
    
    # Also keep the full mapping for quick lookup
    sum_to_count = dict(zip(unique_sums.tolist(), counts.tolist()))
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # Phase 2: Verify and extend with a larger limit
    # Try progressively larger limits
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Start with L2 = 10^10
    L2 = 10**10
    pow_list_2 = generate_powerful_numbers(L2)
    pow_set_2 = set(pow_list_2)
    n2 = len(pow_list_2)
    
    # Check candidates against the larger set
    for N in candidates:
        if N > 2 * pow_list_2[-1]:
            continue
        count = 0
        pairs = []
        for a in pow_list_2:
            if a > N // 2:
                break
            b = N - a
            if b in pow_set_2:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # If we have time, try an even larger limit
    if remaining > 30:
        L2 = 10**11
        pow_list_2 = generate_powerful_numbers(L2)
        pow_set_2 = set(pow_list_2)
        n2 = len(pow_list_2)
        
        for N in candidates:
            if N > 2 * pow_list_2[-1]:
                continue
            count = 0
            pairs = []
            for a in pow_list_2:
                if a > N // 2:
                    break
                b = N - a
                if b in pow_set_2:
                    count += 1
                    pairs.append([a, b])
            if count > best_count:
                best_count = count
                best_N = N
                best_pairs = pairs
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
    
    # If we still have time, try L2 = 10^12
    if remaining > 40:
        L2 = 10**12
        pow_list_2 = generate_powerful_numbers(L2)
        pow_set_2 = set(pow_list_2)
        n2 = len(pow_list_2)
        
        for N in candidates:
            if N > 2 * pow_list_2[-1]:
                continue
            count = 0
            pairs = []
            for a in pow_list_2:
                if a > N // 2:
                    break
                b = N - a
                if b in pow_set_2:
                    count += 1
                    pairs.append([a, b])
            if count > best_count:
                best_count = count
                best_N = N
                best_pairs = pairs
    
    # Fallback
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
