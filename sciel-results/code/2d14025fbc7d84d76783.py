import time
import numpy as np
import random

_N = 256
_AP_CACHE = None

def _get_aps():
    """Precompute all arithmetic progressions as index lists."""
    global _AP_CACHE
    if _AP_CACHE is None:
        aps = []
        for start in range(_N):
            for step in range(1, _N - start):
                seq = []
                idx = start
                while idx < _N:
                    seq.append(idx)
                    idx += step
                aps.append(seq)
        _AP_CACHE = aps
    return _AP_CACHE

def _get_aps_containing(idx):
    """Return list of APs that contain index idx."""
    if not hasattr(_get_aps_containing, '_cache'):
        _get_aps_containing._cache = [None] * _N
    if _get_aps_containing._cache[idx] is None:
        aps = _get_aps()
        _get_aps_containing._cache[idx] = [ap for ap in aps if idx in ap]
    return _get_aps_containing._cache[idx]

def _greedy_construct():
    """
    Greedily assign signs from position 0 to 255.
    At each step, choose +1 or -1 to minimize the maximum absolute AP sum
    over all APs that include the current position (and are fully assigned).
    """
    n = _N
    signs = np.zeros(n, dtype=np.int64)
    aps = _get_aps()
    
    # For each position, precompute which APs contain it
    aps_containing = [None] * n
    for i in range(n):
        aps_containing[i] = [ap for ap in aps if i in ap]
    
    for i in range(n):
        best_val = None
        best_disc = float('inf')
        
        for val in [1, -1]:
            signs[i] = val
            max_disc = 0
            for ap in aps_containing[i]:
                # Only consider APs where all elements are assigned (indices <= i)
                if max(ap) <= i:
                    s = 0
                    for j in ap:
                        s += signs[j]
                    abs_s = abs(s)
                    if abs_s > max_disc:
                        max_disc = abs_s
                        if max_disc >= best_disc:
                            break
            signs[i] = 0  # reset
            if max_disc < best_disc:
                best_disc = max_disc
                best_val = val
        
        signs[i] = best_val
    
    return signs

def _compute_all_ap_sums(signs):
    """Compute sums for all APs. Returns array of (abs_sum, ap_index)."""
    aps = _get_aps()
    sums = np.zeros(len(aps), dtype=np.int64)
    for k, ap in enumerate(aps):
        s = 0
        for j in ap:
            s += signs[j]
        sums[k] = s
    return sums

def _find_worst_ap(signs):
    """Find the AP with maximum absolute sum. Returns (abs_sum, ap_index)."""
    aps = _get_aps()
    max_abs = 0
    worst_idx = 0
    for k, ap in enumerate(aps):
        s = 0
        for j in ap:
            s += signs[j]
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
            worst_idx = k
    return max_abs, worst_idx

def _local_search(signs, time_limit=50.0, seed=42):
    """
    Local search that focuses on flipping elements in the worst AP.
    Uses a tabu-like mechanism to avoid cycling.
    """
    n = _N
    signs = np.array(signs, dtype=np.int64)
    aps = _get_aps()
    
    # Precompute for each AP which indices it contains (as a set for fast lookup)
    # and for each index, which APs it belongs to
    aps_containing = [None] * n
    for i in range(n):
        aps_containing[i] = [k for k, ap in enumerate(aps) if i in ap]
    
    current_sums = _compute_all_ap_sums(signs)
    current_max = int(np.max(np.abs(current_sums)))
    best_disc = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(seed)
    
    # Tabu list: recently flipped indices
    tabu = {}
    tabu_duration = 5
    
    while time.time() - start_time < time_limit:
        max_abs, worst_idx = _find_worst_ap(signs)
        
        if max_abs < best_disc:
            best_disc = max_abs
            best_signs = signs.copy()
        
        if max_abs <= 2:
            break
        
        worst_ap = aps[worst_idx]
        
        # Try flipping each element in the worst AP
        best_flip_idx = None
        best_flip_delta = 0
        best_new_max = max_abs
        
        for idx in worst_ap:
            if tabu.get(idx, -1000) >= len(tabu) - tabu_duration:
                continue
            
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            # Recompute sums for APs containing this index
            affected_aps = aps_containing[idx]
            for k in affected_aps:
                ap = aps[k]
                current_sums[k] += 2 * (-old_sign)  # flip changes sum by -2*old_sign
            
            # Find new max
            new_max = int(np.max(np.abs(current_sums)))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = idx
                best_flip_delta = new_max - max_abs
            
            # Undo flip
            signs[idx] = old_sign
            for k in affected_aps:
                ap = aps[k]
                current_sums[k] += 2 * old_sign  # undo
        
        if best_flip_idx is not None:
            idx = best_flip_idx
            old_sign = signs[idx]
            signs[idx] = -old_sign
            affected_aps = aps_containing[idx]
            for k in affected_aps:
                current_sums[k] += 2 * (-old_sign)
            tabu[idx] = len(tabu)
            current_max = best_new_max
        else:
            # No improving flip found in worst AP; try a random flip anywhere
            idx = rng.randint(0, n - 1)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            affected_aps = aps_containing[idx]
            for k in affected_aps:
                current_sums[k] += 2 * (-old_sign)
            tabu[idx] = len(tabu)
            current_max = int(np.max(np.abs(current_sums)))
    
    return best_signs.tolist(), best_disc

def construct():
    n = _N
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Greedy construction
    signs = _greedy_construct()
    best_signs = signs.tolist()
    
    # Evaluate initial discrepancy
    aps = _get_aps()
    max_disc = 0
    for ap in aps:
        s = 0
        for j in ap:
            s += signs[j]
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    best_disc = max_disc
    
    # Local search with remaining time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5:
        res_signs, res_disc = _local_search(signs, time_limit=remaining - 3.0, seed=42)
        if res_disc < best_disc:
            best_disc = res_disc
            best_signs = res_signs
    
    return best_signs
