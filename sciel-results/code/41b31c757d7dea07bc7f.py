"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
_DEADLINE = 85.0  # safety margin before 90s kill

def _build_critical_aps():
    """Build a list of arithmetic progressions (start, step, length) that are critical.
    We include all APs that could potentially be the maximum.
    To keep it manageable, we focus on:
    - All APs with length >= 2
    - For each (start, step), the maximal length
    - Plus some shorter ones for coverage
    """
    aps = []
    # For each start position (0-indexed) and step, find all maximal APs
    # and also some representative shorter ones
    for start in range(N):
        for step in range(1, N - start):
            # Maximal length for this (start, step)
            max_len = (N - 1 - start) // step + 1
            if max_len < 2:
                continue
            # Add the maximal AP
            aps.append((start, step, max_len))
            # Add a few shorter ones (every other length, or key lengths)
            # For efficiency, add lengths: max_len, max_len-1, and maybe max_len//2
            for L in [max_len - 1, max_len // 2, max_len // 3]:
                if L >= 2 and L not in (max_len, max_len - 1):
                    aps.append((start, step, L))
    
    # Deduplicate
    seen = set()
    unique_aps = []
    for ap in aps:
        if ap not in seen:
            seen.add(ap)
            unique_aps.append(ap)
    
    return unique_aps

def _build_incidence_matrix(aps):
    """Build a binary incidence matrix M where M[i, j] = 1 if position i is in AP j.
    Returns (M, aps_list) where M is (N, num_aps) int8 array.
    """
    num_aps = len(aps)
    M = np.zeros((N, num_aps), dtype=np.int8)
    for j, (start, step, L) in enumerate(aps):
        for k in range(L):
            pos = start + k * step
            if pos < N:
                M[pos, j] = 1
    return M, aps

def _max_abs_sum(signs, M):
    """Compute max |sum| over all APs. signs is (N,) int array of ±1."""
    sums = M.T @ signs  # (num_aps,)
    return np.max(np.abs(sums))

def _max_abs_sum_fast(signs, M):
    """Same as above but returns the value and the index of the max."""
    sums = M.T @ signs
    abs_sums = np.abs(sums)
    idx = np.argmax(abs_sums)
    return abs_sums[idx], idx

def construct():
    start_time = time.time()
    
    # Build critical APs and incidence matrix
    aps = _build_critical_aps()
    M, aps_list = _build_incidence_matrix(aps)
    num_aps = M.shape[1]
    
    # Precompute: for each position, which APs it belongs to
    # This helps for fast delta evaluation
    # M is (N, num_aps), so M[i] gives the APs containing position i
    
    best_signs = None
    best_val = float('inf')
    
    # Multi-restart simulated annealing
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    num_restarts = 20
    max_iters_per_restart = 50000
    
    for restart in range(num_restarts):
        if time.time() - start_time > _DEADLINE:
            break
        
        # Initialize: random or based on previous best
        if restart == 0:
            signs = np_rng.choice([-1, 1], size=N).astype(np.int8)
        else:
            # Perturb the best found so far
            if best_signs is not None:
                signs = best_signs.copy()
                # Flip k random positions
                k = min(N, max(1, N // (4 + restart // 5)))
                flip_idx = np_rng.choice(N, size=k, replace=False)
                signs[flip_idx] *= -1
            else:
                signs = np_rng.choice([-1, 1], size=N).astype(np.int8)
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cool_rate = 0.9995
        current_val, _ = _max_abs_sum_fast(signs, M)
        
        if current_val < best_val:
            best_val = current_val
            best_signs = signs.copy()
        
        iters = 0
        while iters < max_iters_per_restart:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            # Pick a random position to flip
            i = np_rng.randint(N)
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Delta = sum of contributions from APs containing position i
            # For each AP j containing i: sum_j changes by new_sign - old_sign = 2*new_sign
            # Since old_sign is ±1, new_sign = -old_sign, so delta = -2*old_sign for each such AP
            # But we need to check if this increases or decreases |sum_j|
            
            # Get the APs containing position i
            ap_indices = np.where(M[i] == 1)[0]
            if len(ap_indices) == 0:
                iters += 1
                continue
            
            # Current sums for these APs
            current_sums = M[ap_indices].T @ signs  # (len(ap_indices),)
            # After flip, sums become current_sums - 2*old_sign (since we flip sign at i)
            # Wait: sum_j = sum_{k in AP_j} signs[k]. If we flip sign[i], 
            # new_sum_j = old_sum_j - 2*old_sign[i] (because old_sign[i] was added, now -old_sign[i])
            # Actually: old contribution = old_sign[i], new contribution = -old_sign[i]
            # So new_sum_j = old_sum_j - old_sign[i] + (-old_sign[i]) = old_sum_j - 2*old_sign[i]
            
            delta = -2 * old_sign
            new_sums = current_sums + delta
            
            old_abs = np.abs(current_sums)
            new_abs = np.abs(new_sums)
            
            # The objective is max |sum| over ALL APs, not just these
            # But for a local move, the max can only increase if one of the affected APs
            # was at the max and gets worse, or if a new AP becomes the max
            # For efficiency, let's compute the full new max
            
            # Quick check: if no affected AP is near the current max, the max likely unchanged
            # But to be safe, compute new max
            signs[i] = new_sign
            new_val, _ = _max_abs_sum_fast(signs, M)
            
            # Acceptance
            if new_val <= current_val:
                # Always accept improvement
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_signs = signs.copy()
            else:
                # Accept with probability exp(-(new_val - current_val) / T)
                delta_obj = new_val - current_val
                if delta_obj > 0:
                    prob = np.exp(-delta_obj / T)
                    if np_rng.rand() < prob:
                        current_val = new_val
                    else:
                        signs[i] = old_sign  # revert
                else:
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_signs = signs.copy()
            
            T = max(T_min, T * cool_rate)
            iters += 1
        
        # After each restart, if we've hit the deadline, break
        if time.time() - start_time > _DEADLINE - 5:
            break
    
    # Final polish: local search with only accepting improvements
    if best_signs is not None and time.time() - start_time < _DEADLINE - 3:
        signs = best_signs.copy()
        current_val, _ = _max_abs_sum_fast(signs, M)
        
        # Greedy local search: try flipping each position, keep if improves
        improved = True
        while improved and time.time() - start_time < _DEADLINE - 3:
            improved = False
            for i in range(N):
                if time.time() - start_time > _DEADLINE - 3:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                new_val, _ = _max_abs_sum_fast(signs, M)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
        
        # Second pass with random order
        if time.time() - start_time < _DEADLINE - 2:
            improved = True
            while improved and time.time() - start_time < _DEADLINE - 2:
                improved = False
                order = np_rng.permutation(N)
                for i in order:
                    if time.time() - start_time > _DEADLINE - 2:
                        break
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    new_val, _ = _max_abs_sum_fast(signs, M)
                    if new_val < current_val:
                        current_val = new_val
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_signs = signs.copy()
                    else:
                        signs[i] = old_sign
    
    if best_signs is None:
        # Fallback: random
        signs = np_rng.choice([-1, 1], size=N).astype(np.int8)
        return list(signs)
    
    return list(best_signs)
# EVOLVE-BLOCK-END
