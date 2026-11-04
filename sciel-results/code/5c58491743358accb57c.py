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
    deadline = time.time() + 148  # Safety margin for 150s limit
    
    def best_ratio_for_n(n):
        """Find the ratio |U-U|/|U+U| for a well-separated GP of n terms."""
        for b in [1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]:
            U = [0] + [floor(b**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) != n:
                continue
            U_arr = np.array(U, dtype=np.int64)
            sums = np.unique(U_arr[:, None] + U_arr[None, :])
            diffs = np.unique(U_arr[:, None] - U_arr[None, :])
            ratio = len(diffs) / len(sums)
            return ratio
        return 1.0
    
    # Precompute ratios for n = 2..18
    ratios = {}
    for n in range(2, 19):
        ratios[n] = best_ratio_for_n(n)
    
    best_score = -1
    best_set = [0, 1]
    
    # Phase 1: Geometric search for each n
    for n in range(2, 19):
        if time.time() > deadline - 30:
            break
        ratio = ratios[n]
        if ratio <= 1:
            continue
        
        b_lo = 1.0001
        b_hi = 10.0
        
        for _ in range(55):
            if time.time() > deadline - 30:
                break
            b_mid = (b_lo + b_hi) / 2
            U = [0] + [floor(b_mid**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) == n:
                b_hi = b_mid
            else:
                b_lo = b_mid
        
        # Try a range of bases around the minimum
        for margin in [1.0005, 1.001, 1.002, 1.005, 1.01, 1.02, 1.05, 1.1]:
            if time.time() > deadline - 25:
                break
            b_opt = b_hi * margin
            U = [0] + [floor(b_opt**k) for k in range(1, n)]
            U = sorted(set(U))
            if len(U) >= 2:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Phase 2: Local search - perturb individual elements
    if time.time() < deadline - 15:
        current = list(best_set)
        current_score = best_score
        max_u = max(current)
        
        # Generate candidate moves: adjust each element by small amounts
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            
            improved = False
            n_elem = len(current)
            
            for i in range(1, n_elem):  # Skip 0
                orig_val = current[i]
                # Try perturbations: -delta, +delta for various deltas
                deltas = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
                # Scale deltas relative to element value
                if orig_val > 100:
                    deltas = [d * max(1, orig_val // 100) for d in [1, 2, 3, 5, 8, 13, 21]]
                
                for d in deltas:
                    for sign in [-1, 1]:
                        new_val = orig_val + sign * d
                        if new_val < 0:
                            continue
                        # Ensure uniqueness and ordering
                        test_set = current[:i] + [new_val] + current[i+1:]
                        test_set_sorted = sorted(test_set)
                        if len(test_set_sorted) != n_elem:
                            continue
                        if 0 not in test_set_sorted:
                            continue
                        # Check max doesn't explode
                        if max(test_set_sorted) > 3 * max_u + 100:
                            continue
                        
                        s = score_set(test_set_sorted)
                        if s > current_score + 1e-12:
                            current = test_set_sorted
                            current_score = s
                            improved = True
                            max_u = max(current)
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # No improvement in this pass; try larger perturbations
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    # Phase 3: Fine-grained base search for the best n found
    if time.time() < deadline - 5:
        # Determine best n from current best_set
        best_n = len(best_set)
        
        # Binary search for minimum base
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
        
        # Fine search in [b_hi, b_hi * 1.15]
        search_lo = b_hi
        search_hi = b_hi * 1.15
        num_steps = 300
        for i in range(num_steps):
            if time.time() > deadline - 3:
                break
            b = search_lo + (search_hi - search_lo) * i / (num_steps - 1)
            U = [0] + [floor(b**k) for k in range(1, best_n)]
            U = sorted(set(U))
            if len(U) >= 2:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
        
        # Even finer search around the best found
        if best_set and len(best_set) >= 2:
            # Re-estimate base from current best_set
            if len(best_set) >= 3:
                # Estimate base from ratios
                ratios_est = [best_set[k+1]/best_set[k] for k in range(1, min(5, len(best_set)-1)) if best_set[k] > 0]
                if ratios_est:
                    est_b = sum(ratios_est) / len(ratios_est)
                    # Search in a tight window around est_b
                    fine_lo = est_b * 0.98
                    fine_hi = est_b * 1.02
                    for i in range(200):
                        if time.time() > deadline - 2:
                            break
                        b = fine_lo + (fine_hi - fine_lo) * i / 199.0
                        U = [0] + [floor(b**k) for k in range(1, best_n)]
                        U = sorted(set(U))
                        if len(U) >= 2:
                            s = score_set(U)
                            if s > best_score:
                                best_score = s
                                best_set = list(U)
    
    # Phase 4: Test known good sets
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
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 16],
        [0, 1, 2, 6, 18],
        [0, 1, 3, 10, 30],
        [0, 1, 5, 25],
    ]
    for ts in test_sets:
        if time.time() > deadline - 1:
            break
        s = score_set(ts)
        if s > best_score:
            best_score = s
            best_set = list(ts)
    
    # Final local polish
    if time.time() < deadline - 1 and len(best_set) >= 3:
        current = list(best_set)
        current_score = best_score
        for _ in range(200):
            if time.time() > deadline - 0.5:
                break
            improved = False
            n_elem = len(current)
            for i in range(1, n_elem):
                orig_val = current[i]
                for d in [1, 2, 3, 5]:
                    for sign in [-1, 1]:
                        new_val = orig_val + sign * d
                        if new_val < 0:
                            continue
                        test_set = current[:i] + [new_val] + current[i+1:]
                        test_set_sorted = sorted(test_set)
                        if len(test_set_sorted) != n_elem or 0 not in test_set_sorted:
                            continue
                        s = score_set(test_set_sorted)
                        if s > current_score + 1e-12:
                            current = test_set_sorted
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    return best_set
