import time
import math
import random
import numpy as np

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2."""
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

def _update_ap_pure_python(ap, signs, ap_idx, max_sums, min_sums):
    """
    Efficiently update the max and min subarray sums for a specific AP.
    Uses Kadane's algorithm variant for both max and min.
    """
    max_sub = 0
    min_sub = 0
    cur = 0
    min_p = 0
    max_p = 0
    
    # Local variables for speed
    s = signs
    for idx in ap:
        cur += s[idx]
        
        # Update max subarray
        # max_sub is max(cur - min_p)
        # min_p is min(0, cur, ...)
        if cur < min_p:
            min_p = cur
        diff_max = cur - min_p
        if diff_max > max_sub:
            max_sub = diff_max
            
        # Update min subarray
        # min_sub is min(cur - max_p)
        # max_p is max(0, cur, ...)
        if cur > max_p:
            max_p = cur
        diff_min = cur - max_p
        if diff_min < min_sub:
            min_sub = diff_min

    max_sums[ap_idx] = max_sub
    min_sums[ap_idx] = min_sub

def _compute_ap_extrema(aps, signs):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        _update_ap_pure_python(ap, signs, ap_idx, max_sums, min_sums)
        
    return max_sums, min_sums

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    max_neg = int(-np.min(min_sums))
    return max(max_pos, max_neg, 1)

def _find_worst_ap_and_element(aps, signs, max_sums, min_sums):
    """
    Find the AP with the worst discrepancy and a promising element to flip within it.
    Returns (ap_idx, elem_idx, new_disc_estimate) or (None, None, None) if no improvement found.
    """
    n_aps = len(aps)
    if n_aps == 0:
        return None, None, None
    
    # Find AP with max positive subarray sum
    max_pos_ap = int(np.argmax(max_sums))
    max_pos_val = int(max_sums[max_pos_ap])
    
    # Find AP with min negative subarray sum (most negative)
    min_neg_ap = int(np.argmin(min_sums))
    min_neg_val = int(min_sums[min_neg_ap])
    
    # Choose the worse one
    if max_pos_val >= -min_neg_val:
        worst_ap = max_pos_ap
        worst_val = max_pos_val
    else:
        worst_ap = min_neg_ap
        worst_val = min_neg_val
    
    if worst_val <= 1:
        return None, None, None
    
    ap = aps[worst_ap]
    if len(ap) == 0:
        return None, None, None
    
    # Try flipping each element in the worst AP and see which gives best improvement
    # We'll do a quick scan: for each element, compute what the new discrepancy would be
    # by updating just this AP and checking if the global max decreases.
    
    best_elem = None
    best_new_disc = worst_val  # current worst
    
    # Only consider elements in the worst AP
    for elem in ap:
        old_val = signs[elem]
        signs[elem] = -old_val
        
        # Update just this AP
        _update_ap_pure_python(ap, signs, worst_ap, max_sums, min_sums)
        
        # Quick check: does the global max decrease?
        new_max_pos = int(np.max(max_sums))
        new_max_neg = int(-np.min(min_sums))
        new_disc = max(new_max_pos, new_max_neg, 1)
        
        if new_disc < best_new_disc:
            best_new_disc = new_disc
            best_elem = elem
        
        # Revert
        signs[elem] = old_val
    
    # Revert the last element
    if best_elem is not None:
        # We need to restore max_sums/min_sums for worst_ap
        # Re-compute it from the original signs
        _update_ap_pure_python(ap, signs, worst_ap, max_sums, min_sums)
        return worst_ap, best_elem, best_new_disc
    
    return None, None, None

def _spectral_regularize(signs, n, max_freq=16, threshold=0.5):
    """Apply soft-thresholding to DFT coefficients to suppress high-frequency components."""
    x = np.array(signs, dtype=np.float64)
    X = np.fft.rfft(x)
    
    for k in range(1, len(X)):
        if k > max_freq:
            X[k] = np.sign(X[k]) * max(0, abs(X[k]) - threshold)
            
    x_recon = np.fft.irfft(X, n=n)
    signs_out = np.where(x_recon >= 0, 1, -1).astype(int)
    return list(signs_out)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # 1. Build AP Structure
    aps, ap_membership = _build_ap_structure(n)
    n_aps = len(aps)
    
    # 2. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    signs_tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_alt)
    
    # Van der Corput (base-2 digit reversal) - low discrepancy on structured progressions
    signs_vdc = [1 if bin(i ^ (n-1-i)).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_vdc)
    
    # Mixed Thue-Morse with position-based perturbation
    for shift in [1, 2, 3]:
        signs_shifted = [1 if bin((i + shift) % n).count('1') % 2 == 0 else -1 for i in range(n)]
        candidates.append(signs_shifted)
    
    # Random
    for seed in range(5):
        rng_c = random.Random(seed)
        candidates.append([rng_c.choice((-1, 1)) for _ in range(n)])
        
    # Spectral Regularization
    for seed in range(5):
        rng_c = random.Random(seed + 100)
        init = [rng_c.choice((-1, 1)) for _ in range(n)]
        for mf in [8, 16, 32]:
            for th in [0.3, 0.5]:
                reg_signs = _spectral_regularize(init, n, mf, th)
                candidates.append(reg_signs)
    
    # Evaluate candidates to find best start
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 20:
            break
        # Quick check: compute full discrepancy
        max_sums, min_sums = _compute_ap_extrema(aps, c)
        disc = _get_max_disc(max_sums, min_sums)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
            
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    # 3. Phase 1: Steepest Descent Local Search
    # Try all 256 flips, pick the best one. Repeat until no improvement.
    phase1_deadline = deadline - 20
    while time.time() < phase1_deadline:
        improved = False
        best_flip_idx = -1
        best_flip_val = current_disc
        best_new_max_sums = None
        best_new_min_sums = None
        
        for i in range(n):
            old_val = signs[i]
            signs[i] = -old_val
            
            # Update affected APs
            # We need to save the old state of affected APs to revert if this isn't the best
            # Or, since we are just evaluating, we can update, check, and revert.
            # Optimization: Only update if we potentially improve?
            # No, we must update to know the new disc.
            
            affected_aps = ap_membership[i]
            # Save old values for affected APs
            old_max_vals = [max_sums[ap_idx] for ap_idx in affected_aps]
            old_min_vals = [min_sums[ap_idx] for ap_idx in affected_aps]
            
            for ap_idx in affected_aps:
                _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            
            if new_disc < best_flip_val:
                best_flip_val = new_disc
                best_flip_idx = i
                # We don't copy the whole arrays, we just remember the index.
                # We will revert all, then apply the best one.
                
            # Revert
            signs[i] = old_val
            for j, ap_idx in enumerate(affected_aps):
                max_sums[ap_idx] = old_max_vals[j]
                min_sums[ap_idx] = old_min_vals[j]
                
        if best_flip_idx != -1:
            # Apply the best flip found
            i = best_flip_idx
            signs[i] = -signs[i]
            affected_aps = ap_membership[i]
            for ap_idx in affected_aps:
                _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            current_disc = best_flip_val
            improved = True
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            break # No improvement found
            
    # 4. Phase 2a: Targeted Worst-AP Descent
    # Directly attack the highest-discrepancy AP by flipping elements within it
    phase2a_deadline = deadline - 25
    while time.time() < phase2a_deadline:
        ap_idx, elem_idx, est_new_disc = _find_worst_ap_and_element(aps, signs, max_sums, min_sums)
        
        if ap_idx is None:
            break  # No improvement possible via worst-AP targeting
        
        # Apply the flip
        signs[elem_idx] = -signs[elem_idx]
        affected_aps = ap_membership[elem_idx]
        for a_idx in affected_aps:
            _update_ap_pure_python(aps[a_idx], signs, a_idx, max_sums, min_sums)
        
        current_disc = _get_max_disc(max_sums, min_sums)
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
    
    # 4. Phase 2b: Simulated Annealing
    rng_sa = random.Random(123)
    temp = 5.0
    cool_rate = 0.995
    min_temp = 0.01
    no_improve_sa = 0
    max_no_improve_sa = 200
    
    phase2_deadline = deadline - 10
    
    while time.time() < phase2_deadline and no_improve_sa < max_no_improve_sa:
        # Decide: single flip or pair flip?
        if rng_sa.random() < 0.2:
            # Pair flip
            idx1 = rng_sa.randint(0, n - 1)
            idx2 = rng_sa.randint(0, n - 1)
            while idx2 == idx1:
                idx2 = rng_sa.randint(0, n - 1)
            
            old_val1 = signs[idx1]
            old_val2 = signs[idx2]
            
            signs[idx1] = -old_val1
            signs[idx2] = -old_val2
            
            # Update union of affected APs
            affected_aps = set(ap_membership[idx1])
            affected_aps.update(ap_membership[idx2])
            affected_list = list(affected_aps)
            
            old_max_vals = [max_sums[ap_idx] for ap_idx in affected_list]
            old_min_vals = [min_sums[ap_idx] for ap_idx in affected_list]
            
            for ap_idx in affected_list:
                _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            delta = new_disc - current_disc
            
            if delta < 0 or rng_sa.random() < math.exp(-delta / max(temp, 0.01)):
                current_disc = new_disc
                no_improve_sa = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                # Revert
                signs[idx1] = old_val1
                signs[idx2] = old_val2
                for j, ap_idx in enumerate(affected_list):
                    max_sums[ap_idx] = old_max_vals[j]
                    min_sums[ap_idx] = old_min_vals[j]
                no_improve_sa += 1
        else:
            # Single flip
            idx = rng_sa.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            affected_aps = ap_membership[idx]
            old_max_vals = [max_sums[ap_idx] for ap_idx in affected_aps]
            old_min_vals = [min_sums[ap_idx] for ap_idx in affected_aps]
            
            for ap_idx in affected_aps:
                _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            delta = new_disc - current_disc
            
            if delta < 0 or rng_sa.random() < math.exp(-delta / max(temp, 0.01)):
                current_disc = new_disc
                no_improve_sa = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                signs[idx] = old_val
                for j, ap_idx in enumerate(affected_aps):
                    max_sums[ap_idx] = old_max_vals[j]
                    min_sums[ap_idx] = old_min_vals[j]
                no_improve_sa += 1
                
        temp *= cool_rate
        if temp < min_temp:
            temp = min_temp
            
    # 5. Phase 3: Final Steepest Descent Polish (tighter, with worst-AP targeting)
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            best_flip_idx = -1
            best_flip_val = current_disc
            
            # First try worst-AP targeting
            ap_idx, elem_idx, est_new_disc = _find_worst_ap_and_element(aps, signs, max_sums, min_sums)
            if elem_idx is not None:
                best_flip_idx = elem_idx
                best_flip_val = est_new_disc
            
            # Then try all 256 flips
            for i in range(n):
                old_val = signs[i]
                signs[i] = -old_val
                
                affected_aps = ap_membership[i]
                old_max_vals = [max_sums[ap_idx] for ap_idx in affected_aps]
                old_min_vals = [min_sums[ap_idx] for ap_idx in affected_aps]
                
                for ap_idx in affected_aps:
                    _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                    
                new_disc = _get_max_disc(max_sums, min_sums)
                
                if new_disc < best_flip_val:
                    best_flip_val = new_disc
                    best_flip_idx = i
                    
                # Revert
                signs[i] = old_val
                for j, ap_idx in enumerate(affected_aps):
                    max_sums[ap_idx] = old_max_vals[j]
                    min_sums[ap_idx] = old_min_vals[j]
                    
            if best_flip_idx != -1:
                i = best_flip_idx
                signs[i] = -signs[i]
                affected_aps = ap_membership[i]
                for ap_idx in affected_aps:
                    _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                current_disc = _get_max_disc(max_sums, min_sums)
                improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            improved = False
            best_flip_idx = -1
            best_flip_val = current_disc
            
            for i in range(n):
                old_val = signs[i]
                signs[i] = -old_val
                
                affected_aps = ap_membership[i]
                old_max_vals = [max_sums[ap_idx] for ap_idx in affected_aps]
                old_min_vals = [min_sums[ap_idx] for ap_idx in affected_aps]
                
                for ap_idx in affected_aps:
                    _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                    
                new_disc = _get_max_disc(max_sums, min_sums)
                
                if new_disc < best_flip_val:
                    best_flip_val = new_disc
                    best_flip_idx = i
                    
                # Revert
                signs[i] = old_val
                for j, ap_idx in enumerate(affected_aps):
                    max_sums[ap_idx] = old_max_vals[j]
                    min_sums[ap_idx] = old_min_vals[j]
                    
            if best_flip_idx != -1:
                i = best_flip_idx
                signs[i] = -signs[i]
                affected_aps = ap_membership[i]
                for ap_idx in affected_aps:
                    _update_ap_pure_python(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                current_disc = best_flip_val
                improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]

    return best_signs
