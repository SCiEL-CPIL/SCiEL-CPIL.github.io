import time
from math import log, floor
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # For a well-separated geometric progression, |U-U|/|U+U| depends only on
    # the number of distinct elements n, not on the actual values.
    # We precompute the best ratio for each n by finding a base b that makes
    # floor(b^k) all distinct and well-separated.
    
    # Compute best ratio for each n using a "safe" base
    def best_ratio_for_n(n):
        """Find the ratio |U-U|/|U+U| for a well-separated GP of n terms."""
        # Try several bases to ensure distinctness
        for b in [1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]:
            U = [0] + [floor(b**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) != n:
                continue  # not all distinct
            U_arr = np.array(U, dtype=np.int64)
            sums = np.unique(U_arr[:, None] + U_arr[None, :])
            diffs = np.unique(U_arr[:, None] - U_arr[None, :])
            ratio = len(diffs) / len(sums)
            return ratio
        return 1.0
    
    # Precompute ratios for n = 2..15
    ratios = {}
    for n in range(2, 16):
        ratios[n] = best_ratio_for_n(n)
    
    # For each n, the optimal base minimizes max(U) = floor(b^(n-1))
    # subject to all floor(b^k) being distinct and well-separated.
    # The minimal b that ensures distinctness is slightly above 1.
    # For large n, b must be > 1 to avoid collisions.
    # The key tradeoff: smaller b -> smaller max -> smaller D -> better score.
    # But b must be large enough for distinctness.
    
    best_score = -1
    best_set = [0, 1]
    
    for n in range(2, 16):
        if time.time() > deadline - 10:
            break
        ratio = ratios[n]
        if ratio <= 1:
            continue
        
        # Binary search for the smallest b such that floor(b^k) for k=0..n-1
        # are all distinct and the set has exactly n elements.
        # We also want them well-separated to avoid sum/diff collisions.
        
        b_lo = 1.0001
        b_hi = 10.0
        
        # First, find the minimum b for which all n elements are distinct
        # and the set size is exactly n.
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            b_mid = (b_lo + b_hi) / 2
            U = [0] + [floor(b_mid**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) == n:
                b_hi = b_mid
            else:
                b_lo = b_mid
        
        # Now b_hi is approximately the smallest base for distinctness.
        # Use a small margin above b_hi to ensure robustness.
        b_opt = b_hi * 1.001
        
        # Compute the actual set and score
        U = [0] + [floor(b_opt**k) for k in range(1, n)]
        U = sorted(set(U))
        if len(U) != n:
            # Try slightly larger base
            b_opt = b_hi * 1.01
            U = [0] + [floor(b_opt**k) for k in range(1, n)]
            U = sorted(set(U))
        
        if len(U) >= 2:
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Refinement: for the best n, do a fine-grained search over b
    if time.time() < deadline - 5:
        # Find which n gave the best score
        best_n = None
        best_s_for_n = -1
        for n in range(2, 16):
            ratio = ratios.get(n, 1)
            if ratio <= 1:
                continue
            # Reconstruct the minimal-b set for this n
            b_lo = 1.0001
            b_hi = 10.0
            for _ in range(50):
                b_mid = (b_lo + b_hi) / 2
                U = [0] + [floor(b_mid**k) for k in range(1, n)]
                U = sorted(set(U))
                if len(U) == n:
                    b_hi = b_mid
                else:
                    b_lo = b_mid
            
            b_opt = b_hi * 1.001
            U = [0] + [floor(b_opt**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) >= 2:
                s = score_set(U)
                if s > best_s_for_n:
                    best_s_for_n = s
                    best_n = n
        
        if best_n is not None:
            # Fine search: try bases very close to the minimum
            b_lo = 1.0001
            b_hi = 10.0
            for _ in range(60):
                b_mid = (b_lo + b_hi) / 2
                U = [0] + [floor(b_mid**k) for k in range(1, best_n)]
                U = sorted(set(U))
                if len(U) == best_n:
                    b_hi = b_mid
                else:
                    b_lo = b_mid
            
            # Search in [b_hi, b_hi * 1.1] for the best score
            search_lo = b_hi
            search_hi = b_hi * 1.1
            for i in range(200):
                if time.time() > deadline - 2:
                    break
                b = search_lo + (search_hi - search_lo) * i / 199.0
                U = [0] + [floor(b**k) for k in range(1, best_n)]
                U = sorted(set(U))
                if len(U) >= 2:
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Also try the known good small sets
    test_sets = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 7, 15],
    ]
    for ts in test_sets:
        if time.time() > deadline - 1:
            break
        s = score_set(ts)
        if s > best_score:
            best_score = s
            best_set = list(ts)
    
    return best_set
