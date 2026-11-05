import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or numpy array of +1/-1.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    # Iterate over step d
    for d in range(1, n + 1):
        # For each residue class r in 0..d-1
        # We only care if the AP has at least 2 elements? 
        # The problem says "length including one". 
        # A singleton has sum +/-1, abs 1.
        # So max_abs is at least 1.
        for r in range(min(d, n)):
            # Extract subsequence: indices r, r+d, r+2d, ...
            # np.arange(r, n, d)
            # Using slicing is faster: signs_arr[r::d]
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                # Sum is +/-1, abs is 1
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            # Calculate max subarray sum and min subarray sum
            # cumsum
            cum = np.cumsum(sub)
            
            # To find max subarray sum:
            # max_{i,j, i<j} (cum[j] - cum[i])
            # This is max(cum[j]) - min(cum[i]) where i < j.
            # We can use minimum.accumulate
            # Prepend 0 to cum to allow subarray starting at 0
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            # min_prefix[i] is min(cum_ext[0..i])
            
            # Max subarray ending at j (1-indexed in cum_ext, 0-indexed in cum)
            # diff = cum[j] - min(cum[0..j-1])
            # Note: cum_ext has length m+1. cum has length m.
            # cum[j] corresponds to cum_ext[j+1].
            # We want max over j=0..m-1 of cum_ext[j+1] - min_prefix[j]
            
            # Vectorized:
            # diffs = cum_ext[1:] - min_prefix[:-1]
            # max_sub = np.max(diffs)
            
            # Similarly for min subarray:
            # min_{i<j} (cum[j] - cum[i]) = min(cum[j]) - max(cum[i])
            max_prefix = np.maximum.accumulate(cum_ext)
            # diffs_min = cum_ext[1:] - max_prefix[:-1]
            # min_sub = np.min(diffs_min)
            
            # Combine
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                # Early exit? If max_abs is already high, can't exit easily 
                # unless we have a bound, but we are minimizing, so we need exact max.
                
    return max_abs

def _build_ap_structure(n):
    """
    Precompute all arithmetic progressions (start, step, length>=2) 
    and index their elements for fast incremental discrepancy tracking.
    Returns:
        aps: list of lists of global indices
        ap_membership: list of lists where ap_membership[i] = list of AP indices that include position i
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    return aps, ap_membership

def _compute_ap_extrema(aps, signs):
    """
    For each AP, compute the maximum subarray sum and minimum subarray sum.
    Returns lists: max_sums, min_sums.
    """
    max_sums = []
    min_sums = []
    for ap in aps:
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        
        max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
        
        max_sums.append(max_sub)
        min_sums.append(min_sub)
    
    return max_sums, min_sums

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """
    Recompute max/min subarray sums for a single AP.
    """
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _get_max_disc(max_sums, min_sums):
    """Get the maximum absolute discrepancy from precomputed sums."""
    if not max_sums:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    tm = []
    for i in range(n):
        cnt = bin(i).count('1')
        tm.append(1 if cnt % 2 == 0 else -1)
    candidates.append(tm)
    
    # Walsh-Hadamard (Rademacher functions)
    for k in [1, 2, 3, 4]:
        wh = []
        for i in range(n):
            wh.append(1 if (i >> k) & 1 == 0 else -1)
        candidates.append(wh)
        
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
        
    # Sine wave
    for freq in [0.5, 0.7, 1.0, 1.5, 2.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)

    # Evaluate and pick best
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]

    # 2. Local Search with Incremental Discrepancy Tracking
    signs = best_signs[:]
    
    # Compute initial AP extrema
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 50
    
    # Simulated annealing temperature
    temp = 5.0
    cool_rate = 0.95
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try multiple random flips per iteration
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            new_val = -old_val
            
            # Flip the sign
            signs[idx] = new_val
            
            # Update all APs that include this index
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            
            new_disc = _get_max_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                # Accept improvement
                current_disc = new_disc
                improved = True
                no_improve_count = 0
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                # Simulated annealing: accept worse moves with probability exp(-delta/temp)
                delta = new_disc - current_disc
                if delta > 0 and temp > 0.1:
                    accept_prob = math.exp(-delta / temp)
                    if rng.random() < accept_prob:
                        current_disc = new_disc
                        no_improve_count = 0
                    else:
                        # Revert
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                else:
                    # Revert
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        if not improved:
            no_improve_count += 1
        
        # Cool down
        temp *= cool_rate
        
        # If stuck for too long, do a random restart perturbation
        if no_improve_count > max_no_improve:
            # Perturb several random positions
            for _ in range(5):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
            
            # Recompute all AP extrema
            max_sums, min_sums = _compute_ap_extrema(aps, signs)
            current_disc = _get_max_disc(max_sums, min_sums)
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
            
            no_improve_count = 0
            temp = max(temp, 2.0)  # Reheat

    # Final verification
    final_disc = _max_ap_discrepancy(signs)
    if final_disc < best_disc:
        best_disc = final_disc
        best_signs = signs[:]
    
    return best_signs
