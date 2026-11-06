import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    L = 10**9
    
    # Generate powerful numbers up to L
    max_v = int(L ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_v
    is_sq_free = np.ones(max_v + 1, dtype=bool)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for p in range(2, isqrt(max_v) + 1):
        if is_sq_free[p]:
            p2 = p * p
            is_sq_free[p2::p2] = False
            
    square_free_v = np.nonzero(is_sq_free)[0]
    square_free_v = square_free_v[square_free_v**3 <= L]
    
    powerfuls = set()
    powerfuls.add(1)
    
    for v in square_free_v:
        v3 = v**3
        max_u = isqrt(L // v3)
        if max_u < 1:
            continue
        u_vals = np.arange(1, max_u + 1)
        n_vals = (u_vals**2) * v3
        for n in n_vals:
            powerfuls.add(int(n))
            
    powerfuls_list = np.array(sorted(powerfuls), dtype=np.int64)
    P = len(powerfuls_list)
    p_set = set(powerfuls_list.tolist())
    
    # Phase 1: Find candidate N using pairwise sums of a subset of powerful numbers
    # Take powerful numbers in [L//4, L] as "central"
    central_mask = (powerfuls_list >= L // 4) & (powerfuls_list <= L)
    central = powerfuls_list[central_mask]
    
    # Also include small powerful numbers (1, 4, 8, 9, 16, ...) for diversity
    small_mask = powerfuls_list <= 10000
    small = powerfuls_list[small_mask]
    
    # Combine: sample from central + all small
    if len(central) > 2500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(central), size=2500, replace=False)
        central_sample = central[idx]
    else:
        central_sample = central
    
    combined = np.concatenate([central_sample, small])
    cs = len(combined)
    
    # Compute all pairwise sums (i <= j)
    n_sums = cs * (cs + 1) // 2
    sums = np.empty(n_sums, dtype=np.int64)
    pos = 0
    for i in range(cs):
        sums[pos:pos + cs - i] = combined[i] + combined[i:]
        pos += cs - i
    
    # Find the sum with highest multiplicity
    unique_sums, counts = np.unique(sums, return_counts=True)
    best_idx = np.argmax(counts)
    candidate_N = int(unique_sums[best_idx])
    
    # Phase 2: For candidate_N and neighbors, do full count over all powerful numbers
    best_N = candidate_N
    best_count = 0
    
    # Check candidate_N and neighbors (even values)
    search_range = 200
    for N in range(candidate_N - search_range, candidate_N + search_range + 1, 2):
        if N < 2 or N > 2 * L:
            continue
        if time.time() > deadline - 5:
            break
        # Count representations: a <= b, a + b = N, both powerful
        count = 0
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in p_set:
                count += 1
        if count > best_count:
            best_count = count
            best_N = N
    
    # Extract pairs for best_N
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set and a <= b:
            pairs.append([int(a), int(b)])
    
    return {'N': best_N, 'pairs': pairs}
