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
    """Find the AP with maximum absolute discrepancy."""
    if len(max_sums) == 0:
        return -1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    if max_pos >= min_neg:
        return int(np.argmax(max_sums))
    else:
        return int(np.argmin(min_sums))

def _fast_flip_eval(pos, signs, ap_membership, aps, max_sums, min_sums, ap_lengths):
    """Fast evaluation of flipping position pos: compute new max discrepancy."""
    old_val = signs[pos]
    signs[pos] = -old_val
    
    affected = ap_membership[pos]
    old_vals = {}
    for ap_idx in affected:
        old_vals[ap_idx] = (max_sums[ap_idx], min_sums[ap_idx])
        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
    
    new_disc = _get_max_disc(max_sums, min_sums)
    
    # Restore
    signs[pos] = old_val
    for ap_idx, (om, omn) in old_vals.items():
        max_sums[ap_idx] = om
        min_sums[ap_idx] = omn
        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
    
    return new_disc

def _find_best_flip_in_ap(ap, signs, ap_membership, aps, max_sums, min_sums, ap_lengths):
    """Find the best position to flip within a specific AP to reduce discrepancy.
    Prioritizes positions with high absolute contribution to the worst AP."""
    n = len(signs)
    best_idx = -1
    best_disc = float('inf')
    
    # Compute contribution of each position in the AP to its sum
    # Positions that are 'misaligned' (sign opposite to the AP's dominant direction)
    # are most likely to reduce discrepancy when flipped
    
    # Get the current sum of the AP
    ap_sum = sum(signs[idx] for idx in ap)
    
    # Sort positions by how much they contribute against the AP's direction
    # If ap_sum > 0, positions with sign -1 are 'misaligned'
    # If ap_sum < 0, positions with sign +1 are 'misaligned'
    if ap_sum >= 0:
        # Prefer flipping -1 positions (they reduce positive sum)
        sorted_pos = sorted(ap, key=lambda p: (signs[p], -abs(signs[p])))
    else:
        # Prefer flipping +1 positions (they reduce negative sum magnitude)
        sorted_pos = sorted(ap, key=lambda p: (-signs[p], abs(signs[p])))
    
    # Try at most 10 positions, starting with most misaligned
    for pos in sorted_pos[:min(10, len(ap))]:
        new_disc = _fast_flip_eval(pos, signs, ap_membership, aps, max_sums, min_sums, ap_lengths)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_idx = pos
    
    # If no improvement found in top 10, try a few random positions
    if best_idx == -1 and len(ap) > 10:
        remaining = sorted_pos[10:]
        random_sample = remaining[:5] if len(remaining) >= 5 else remaining
        for pos in random_sample:
            new_disc = _fast_flip_eval(pos, signs, ap_membership, aps, max_sums, min_sums, ap_lengths)
            if new_disc < best_disc:
                best_disc = new_disc
                best_idx = pos
    
    return best_idx, best_disc

def _greedy_construct(n, aps, ap_membership, ap_lengths, seed=0):
    """Greedy construction: pick each sign to minimize current max discrepancy."""
    rng = random.Random(seed)
    signs = [0] * n
    max_sums = np.zeros(len(aps), dtype=np.int32)
    min_sums = np.zeros(len(aps), dtype=np.int32)
    
    # Initialize all to 0, then greedily assign
    # Start with random initialization
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    return signs

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
    for seed in range(30):
        rng = random.Random(seed)
        candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Greedy-like: start with all +1, then flip to reduce
    greedy1 = [1] * n
    candidates.append(greedy1)
    
    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)
    
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline - 15:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    # Targeted local search: aggressively reduce worst AP discrepancy
    rng = random.Random(42)
    
    for restart in range(8):
        if time.time() > deadline - 5:
            break
        
        if restart > 0:
            # Restart from a perturbed version of best
            signs = best_signs[:]
            # Perturb 5-15% of positions depending on restart
            perturb_count = max(5, n // (10 + restart))
            for _ in range(perturb_count):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
            max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
            current_disc = _get_max_disc(max_sums, min_sums)
        else:
            # Continue from current best
            signs = best_signs[:]
            max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
            current_disc = _get_max_disc(max_sums, min_sums)
        
        no_improve_count = 0
        max_no_improve = 150
        consecutive_targeted_fail = 0
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            # Targeted approach: find worst AP and try flipping positions within it
            worst_ap_idx = _find_worst_ap(max_sums, min_sums)
            if worst_ap_idx >= 0:
                worst_ap = aps[worst_ap_idx]
                # Try flipping positions in the worst AP, prioritizing misaligned signs
                best_flip_idx, best_flip_disc = _find_best_flip_in_ap(
                    worst_ap, signs, ap_membership, aps, max_sums, min_sums, ap_lengths
                )
                
                if best_flip_idx >= 0 and best_flip_disc < current_disc:
                    # Apply the best flip
                    old_val = signs[best_flip_idx]
                    signs[best_flip_idx] = -old_val
                    for ap_idx in ap_membership[best_flip_idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                    current_disc = best_flip_disc
                    no_improve_count = 0
                    consecutive_targeted_fail = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    consecutive_targeted_fail += 1
                    # If targeted approach fails multiple times, try random flips
                    if consecutive_targeted_fail >= 3:
                        # Try 3 random flips
                        improved_this_round = False
                        for _ in range(3):
                            idx = rng.randint(0, n - 1)
                            new_disc = _fast_flip_eval(idx, signs, ap_membership, aps, max_sums, min_sums, ap_lengths)
                            if new_disc < current_disc:
                                # Apply this flip
                                old_val = signs[idx]
                                signs[idx] = -old_val
                                for ap_idx in ap_membership[idx]:
                                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                                current_disc = new_disc
                                improved_this_round = True
                                if current_disc < best_disc:
                                    best_disc = current_disc
                                    best_signs = signs[:]
                                break
                        
                        if improved_this_round:
                            no_improve_count = 0
                            consecutive_targeted_fail = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Final polish: strict hill climbing with targeted priority
        if time.time() < deadline - 2:
            for _ in range(300):
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # First pass: try flipping in worst APs
                worst_ap_idx = _find_worst_ap(max_sums, min_sums)
                if worst_ap_idx >= 0:
                    worst_ap = aps[worst_ap_idx]
                    for pos in worst_ap[:min(5, len(worst_ap))]:
                        new_disc = _fast_flip_eval(pos, signs, ap_membership, aps, max_sums, min_sums, ap_lengths)
                        if new_disc < current_disc:
                            old_val = signs[pos]
                            signs[pos] = -old_val
                            for ap_idx in ap_membership[pos]:
                                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                            current_disc = new_disc
                            improved = True
                            if current_disc < best_disc:
                                best_disc = current_disc
                                best_signs = signs[:]
                            break
                
                # Second pass: try all positions
                if not improved:
                    for idx in range(n):
                        new_disc = _fast_flip_eval(idx, signs, ap_membership, aps, max_sums, min_sums, ap_lengths)
                        if new_disc < current_disc:
                            old_val = signs[idx]
                            signs[idx] = -old_val
                            for ap_idx in ap_membership[idx]:
                                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                            current_disc = new_disc
                            improved = True
                            if current_disc < best_disc:
                                best_disc = current_disc
                                best_signs = signs[:]
                            break
                
                if not improved:
                    break
    
    # Final verification
    final_disc = _max_ap_discrepancy(best_signs)
    if final_disc < best_disc:
        best_disc = final_disc
    
    # One more round of targeted improvement if time permits
    if time.time() < deadline - 1:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
        current_disc = _get_max_disc(max_sums, min_sums)
        
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            worst_ap_idx = _find_worst_ap(max_sums, min_sums)
            if worst_ap_idx < 0:
                break
            worst_ap = aps[worst_ap_idx]
            best_flip_idx, best_flip_disc = _find_best_flip_in_ap(
                worst_ap, signs, ap_membership, aps, max_sums, min_sums, ap_lengths
            )
            if best_flip_idx >= 0 and best_flip_disc < current_disc:
                old_val = signs[best_flip_idx]
                signs[best_flip_idx] = -old_val
                for ap_idx in ap_membership[best_flip_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                current_disc = best_flip_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                break
    
    return best_signs
