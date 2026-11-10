"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_aps(n):
    """
    Precompute all arithmetic progressions in [1, n].
    For each position i (1-indexed), store list of (ap_id, index_in_ap)
    where ap_id identifies the AP and index_in_ap is the position within it.
    
    Returns:
        ap_indices: list of lists, ap_indices[ap_id] = list of 1-indexed positions
        pos_to_aps: list of lists, pos_to_aps[i] = list of (ap_id, idx_in_ap)
    """
    ap_indices = []
    pos_to_aps = [[] for _ in range(n + 1)]
    
    for d in range(1, n + 1):
        for s in range(1, n + 1):
            # Build the AP: s, s+d, s+2d, ...
            ap = []
            pos = s
            while pos <= n:
                ap.append(pos)
                pos += d
            if len(ap) >= 1:  # Include singletons
                ap_id = len(ap_indices)
                ap_indices.append(ap)
                for idx, p in enumerate(ap):
                    pos_to_aps[p].append((ap_id, idx))
    
    return ap_indices, pos_to_aps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    rng = random.Random(42)
    
    # Precompute APs
    ap_indices, pos_to_aps = _precompute_aps(n)
    num_aps = len(ap_indices)
    
    # Initialize signs (1-indexed internally, index 0 unused)
    signs = np.zeros(n + 1, dtype=np.int8)
    for i in range(1, n + 1):
        signs[i] = rng.choice([-1, 1])
    
    # Compute initial AP sums
    # ap_sums[ap_id] = sum of signs over that AP
    ap_sums = np.zeros(num_aps, dtype=np.int16)
    for ap_id, ap in enumerate(ap_indices):
        total = 0
        for p in ap:
            total += signs[p]
        ap_sums[ap_id] = total
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99999
    
    while time.time() < deadline:
        # Pick random position to flip
        i = rng.randint(1, n)
        old_sign = signs[i]
        new_sign = -old_sign
        signs[i] = new_sign
        
        # Update AP sums that contain position i
        # For each AP containing i, the sum changes by 2*new_sign (since we flipped)
        # Actually, change = new_sign - old_sign = 2*new_sign (since old_sign = -new_sign)
        delta = 2 * new_sign  # This is the change in each AP sum that contains i
        
        for ap_id, idx_in_ap in pos_to_aps[i]:
            ap_sums[ap_id] += delta
        
        # Compute new max
        new_max = int(np.max(np.abs(ap_sums)))
        delta_score = new_max - current_max
        
        if delta_score <= 0:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > T_min:
                prob = np.exp(-delta_score / T)
                if rng.random() < prob:
                    current_max = new_max
                else:
                    # Reject: revert
                    signs[i] = old_sign
                    for ap_id, idx_in_ap in pos_to_aps[i]:
                        ap_sums[ap_id] -= delta
            else:
                # Reject at low temperature
                signs[i] = old_sign
                for ap_id, idx_in_ap in pos_to_aps[i]:
                    ap_sums[ap_id] -= delta
        
        T *= cooling
        
        if T < T_min:
            # Reheat with some probability, restart from best
            if rng.random() < 0.3:
                T = 5.0
                signs[:] = best_signs
                # Recompute ap_sums from best_signs
                for ap_id, ap in enumerate(ap_indices):
                    total = 0
                    for p in ap:
                        total += best_signs[p]
                    ap_sums[ap_id] = total
                current_max = best_max
    
    return best_signs[1:].tolist()
# EVOLVE-BLOCK-END
