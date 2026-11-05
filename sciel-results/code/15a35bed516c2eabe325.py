import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1,n]."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            cum = np.cumsum(sub)
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2."""
    aps = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                ap_lengths.append(len(sub_indices))
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    return aps, ap_membership, ap_lengths

def _compute_ap_extrema(aps, signs, ap_lengths):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        _update_single_ap(ap, signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
        
    return max_sums, min_sums

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums, length):
    """Update max/min subarray sum for a single AP."""
    if length < 50:
        max_end = 0
        max_sum = 0
        min_end = 0
        min_sum = 0
        
        for idx in ap:
            v = signs[idx]
            max_end = max(0, max_end + v)
            if max_end > max_sum:
                max_sum = max_end
            
            min_end = max(0, min_end - v)
            if min_end > min_sum:
                min_sum = min_end
                
        max_sums[ap_idx] = max_sum
        min_sums[ap_idx] = -min_sum
    else:
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        
        max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _find_worst_ap(max_sums, min_sums):
    """Find the AP index with the maximum discrepancy."""
    if len(max_sums) == 0:
        return 0
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    
    if max_pos >= min_neg:
        return int(np.argmax(max_sums))
    else:
        return int(np.argmin(min_sums))

def _greedy_worst_ap_repair(signs, aps, ap_membership, ap_lengths, max_sums, min_sums, deadline, rng):
    """Greedy repair: repeatedly find worst AP and flip a sign to reduce it."""
    current_disc = _get_max_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        worst_idx = _find_worst_ap(max_sums, min_sums)
        worst_ap = aps[worst_idx]
        worst_len = ap_lengths[worst_idx]
        
        # Get current sum of the worst AP
        ap_signs = [signs[idx] for idx in worst_ap]
        current_sum = sum(ap_signs)
        
        if current_sum == 0:
            # Find a sign to flip that might help other APs
            flip_idx = rng.choice(worst_ap)
        else:
            # Flip a sign that has the same sign as current_sum
            target_sign = 1 if current_sum > 0 else -1
            candidates = [idx for idx in worst_ap if signs[idx] == target_sign]
            
            if not candidates:
                flip_idx = rng.choice(worst_ap)
            else:
                # Pick randomly among candidates
                flip_idx = rng.choice(candidates)
        
        # Flip the sign
        old_val = signs[flip_idx]
        signs[flip_idx] = -old_val
        
        # Update affected APs
        for ap_idx in ap_membership[flip_idx]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs[:]
            no_improve = 0
        else:
            # Revert
            signs[flip_idx] = old_val
            for ap_idx in ap_membership[flip_idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
            no_improve += 1
            
            # If we can't improve, try a random flip to escape local minimum
            if no_improve % 10 == 0:
                flip_idx = rng.randint(0, len(signs) - 1)
                old_val = signs[flip_idx]
                signs[flip_idx] = -old_val
                for ap_idx in ap_membership[flip_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
        
        current_disc = _get_max_disc(max_sums, min_sums)
    
    return best_signs, best_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 256
    
    aps, ap_membership, ap_lengths = _build_ap_structure(n)
    
    candidates = []
    
    # Thue-Morse sequence
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)
    
    # Multiple random seeds
    for seed in range(20):
        rng = random.Random(seed)
        candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)
    
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline - 20:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    # Phase 1: Greedy Worst-AP Repair
    if time.time() < deadline - 10:
        signs, best_disc = _greedy_worst_ap_repair(signs, aps, ap_membership, ap_lengths, max_sums, min_sums, deadline, random.Random(42))
        best_signs = signs[:]
    
    # Phase 2: Simulated Annealing
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if time.time() < deadline - 5:
        rng = random.Random(123)
        temp = 2.0
        cool_rate = 0.998
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            # Batch of flips
            batch_size = 30
            improved_in_batch = False
            
            for _ in range(batch_size):
                if time.time() > deadline - 3:
                    break
                
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                new_val = -old_val
                
                signs[idx] = new_val
                
                for ap_idx in ap_membership[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                
                new_disc = _get_max_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved_in_batch = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                        no_improve_count = 0
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        accept_prob = math.exp(-delta / temp)
                        if rng.random() < accept_prob:
                            current_disc = new_disc
                        else:
                            signs[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                    else:
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
            
            if not improved_in_batch:
                no_improve_count += 1
            
            temp *= cool_rate
    
    # Phase 3: Final polish with strict hill climbing
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                old_val = signs[idx]
                signs[idx] = -old_val
                
                for ap_idx in ap_membership[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                
                new_disc = _get_max_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
    
    # Final verification
    final_disc = _max_ap_discrepancy(best_signs)
    if final_disc < best_disc:
        best_disc = final_disc
    
    return best_signs
