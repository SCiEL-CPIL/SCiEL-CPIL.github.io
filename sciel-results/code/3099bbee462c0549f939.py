import time
import random
import numpy as np
import math

def _compute_ap_extrema(aps, signs):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])
    return max_sums, min_sums

def _find_worst_aps(max_sums, min_sums, aps, top_k=10):
    """Find the APs with the largest absolute discrepancy."""
    disc = np.maximum(max_sums, -min_sums)
    if len(disc) == 0:
        return []
    worst_indices = np.argsort(disc)[::-1][:top_k]
    return [(int(idx), int(disc[idx])) for idx in worst_indices if disc[idx] > 0]

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _get_sum_squares(max_sums, min_sums):
    """Get the sum of squares of max and min sums (proxy objective)."""
    if len(max_sums) == 0:
        return 0
    return float(np.sum(max_sums**2) + np.sum(min_sums**2))

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """Update max/min subarray sum for a single AP in-place."""
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2 and membership info."""
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

def _sa_run(n, aps, ap_membership, initial_signs, time_limit, rng_seed):
    """Run a single simulated annealing instance with targeted descent."""
    rng = random.Random(rng_seed)
    signs = initial_signs[:]
    
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    current_ss = _get_sum_squares(max_sums, min_sums)
    
    best_disc = current_disc
    best_signs = signs[:]
    best_ss = current_ss
    
    temp = 10.0
    cool_rate = 0.999
    start_time = time.time()
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        # Every so often, do targeted descent on worst APs
        if no_improve > 200:
            # Targeted descent: try flipping elements in worst APs
            worst_aps = _find_worst_aps(max_sums, min_sums, aps, top_k=5)
            for ap_idx, disc_val in worst_aps:
                if time.time() - start_time > time_limit:
                    break
                ap = aps[ap_idx]
                # Try flipping each element in this AP
                for pos in ap:
                    old_val = signs[pos]
                    signs[pos] = -old_val
                    for ap_idx2 in ap_membership[pos]:
                        _update_single_ap(aps[ap_idx2], signs, ap_idx2, max_sums, min_sums)
                    new_disc = _get_max_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        current_ss = _get_sum_squares(max_sums, min_sums)
                    else:
                        signs[pos] = old_val
                        for ap_idx2 in ap_membership[pos]:
                            _update_single_ap(aps[ap_idx2], signs, ap_idx2, max_sums, min_sums)
            
            no_improve = 0
            temp = max(temp, 5.0)
        
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        new_val = -old_val
        
        signs[idx] = new_val
        
        old_ss = current_ss
        old_disc = current_disc
        
        # Update affected APs
        ss_delta = 0.0
        for ap_idx in ap_membership[idx]:
            old_max = max_sums[ap_idx]
            old_min = min_sums[ap_idx]
            
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            
            new_max = max_sums[ap_idx]
            new_min = min_sums[ap_idx]
            
            ss_delta += (new_max**2 + new_min**2) - (old_max**2 + old_min**2)
        
        current_ss = old_ss + ss_delta
        new_disc = _get_max_disc(max_sums, min_sums)
        current_disc = new_disc
        
        delta_ss = current_ss - old_ss
        
        if delta_ss < 0 or rng.random() < np.exp(-delta_ss / max(temp, 0.01)):
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
                best_ss = current_ss
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert
            signs[idx] = old_val
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            current_ss = old_ss
            current_disc = old_disc
            no_improve += 1
            
        temp *= cool_rate
        
        if best_disc == 1:
            break
            
        if no_improve > max_no_improve:
            # Perturb: flip a random block or restart cooling
            temp = 10.0
            no_improve = 0
            
    return best_disc, best_signs

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 256
    
    # Build AP structure once
    aps, ap_membership = _build_ap_structure(n)
    
    best_global_disc = float('inf')
    best_global_signs = None
    
    # Multiple starts with different seeds and initializations
    seeds = [42, 123, 999, 2023, 777, 555, 888]
    
    # Calculate time per run - give more time to first runs
    time_per_run = (deadline - start_time) / len(seeds)
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        # Initialization
        if i == 0:
            # Thue-Morse
            signs = [1 if bin(j).count('1') % 2 == 0 else -1 for j in range(n)]
        elif i == 1:
            # Random balanced
            signs = [1] * (n // 2) + [-1] * (n - n // 2)
            random.Random(seed).shuffle(signs)
        elif i == 2:
            # Anti-Thue-Morse
            signs = [-1 if bin(j).count('1') % 2 == 0 else 1 for j in range(n)]
        else:
            # Pure random
            signs = [1 if random.Random(seed).randint(0, 1) == 0 else -1 for _ in range(n)]
        
        disc, signs_out = _sa_run(n, aps, ap_membership, signs, time_per_run, seed)
        
        if disc < best_global_disc:
            best_global_disc = disc
            best_global_signs = signs_out[:]
            
        if best_global_disc == 1:
            break
            
    # Final polish on the best found
    if best_global_signs is not None and time.time() < deadline - 2:
        # Do a quick greedy polish on the best result
        signs = best_global_signs[:]
        max_sums, min_sums = _compute_ap_extrema(aps, signs)
        current_disc = _get_max_disc(max_sums, min_sums)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                new_disc = _get_max_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        if current_disc < best_global_disc:
            best_global_disc = current_disc
            best_global_signs = signs[:]

    return best_global_signs
