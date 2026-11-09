"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill
    
    # Precompute all arithmetic progressions
    # An AP is defined by start index s (0-indexed) and step d (positive integer)
    # The AP contains indices: s, s+d, s+2d, ... < n
    # We need to track the sum over each such AP
    # For efficiency, we'll store for each index which APs it belongs to,
    # and for each AP, the list of indices in it.
    
    # Actually, for incremental updates, we need:
    # - For each index i, the list of (ap_id, position_in_ap) or just ap_id
    # - For each ap_id, the list of indices
    # When we flip sign at index i, for each AP containing i, the sum changes by 2*sign[i]
    
    # Let's precompute the APs
    aps = []  # list of lists of indices
    index_to_aps = [[] for _ in range(n)]  # for each index, list of ap_ids
    
    ap_id_counter = 0
    for d in range(1, n + 1):
        for start in range(0, n, d):
            indices = list(range(start, n, d))
            if indices:
                ap_id = ap_id_counter
                ap_id_counter += 1
                aps.append(indices)
                for idx in indices:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Initialize with random signs
    rng = random.Random(12345)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    # Compute initial AP sums
    ap_sums = [0] * num_aps
    for ap_id in range(num_aps):
        s = 0
        for idx in aps[ap_id]:
            s += signs[idx]
        ap_sums[ap_id] = s
    
    # Track max absolute sum
    def compute_max_abs_sum(ap_sums):
        return max(abs(x) for x in ap_sums)
    
    current_max = compute_max_abs_sum(ap_sums)
    best_signs = signs[:]
    best_max = current_max
    
    # Simulated annealing with constraint-directed flips
    # Temperature schedule
    T0 = 10.0
    T_min = 0.01
    alpha = 0.999  # cooling rate per step
    
    step = 0
    total_steps = 50000  # max steps, but we'll check time
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if step >= total_steps:
            break
        
        T = T0 * (alpha ** step)
        if T < T_min:
            T = T_min
        
        # Choose a flip target
        # Strategy: with some probability, pick from the worst APs; otherwise random
        # Find APs with maximum absolute sum
        max_abs = current_max
        # Collect indices that are in APs with |sum| == max_abs
        candidate_indices = set()
        for ap_id in range(num_aps):
            if abs(ap_sums[ap_id]) == max_abs:
                for idx in aps[ap_id]:
                    candidate_indices.add(idx)
        
        if candidate_indices and rng.random() < 0.7:
            # Pick a random index from the worst APs
            idx = rng.choice(list(candidate_indices))
        else:
            idx = rng.randint(0, n - 1)
        
        # Try flipping sign at idx
        old_sign = signs[idx]
        new_sign = -old_sign
        
        # Compute delta in max abs sum
        # For each AP containing idx, the sum changes by 2*old_sign (since we flip from old to new, delta = new - old = -2*old)
        # Wait: if we flip from +1 to -1, the sum decreases by 2. If from -1 to +1, increases by 2.
        # delta_sum = new_sign - old_sign = -2 * old_sign
        
        delta = -2 * old_sign
        
        # Compute new max abs sum efficiently
        # We need to check if the new max is lower
        # The new sums for APs containing idx will change
        # The new max is max of:
        #   - old max among APs NOT containing idx
        #   - new |sum| for APs containing idx
        
        # For simplicity, let's just compute the new max by checking all APs
        # But that's O(num_aps) per flip. num_aps is around 256*256/2 ~ 32768? Let me estimate.
        # Actually, for step d, there are n//d + 1 APs (roughly). Sum over d=1 to n of n//d ≈ n*H(n) ≈ 256*6 ≈ 1536.
        # So num_aps is around 1500-2000. That's manageable.
        
        # Apply flip
        signs[idx] = new_sign
        for ap_id in index_to_aps[idx]:
            ap_sums[ap_id] += delta
        
        new_max = compute_max_abs_sum(ap_sums)
        
        # Acceptance criterion
        delta_max = new_max - current_max
        if delta_max <= 0:
            # Improvement or same
            accept = True
        else:
            # Worse; accept with probability exp(-delta_max / T)
            if T > 0:
                accept = (rng.random() < math.exp(-delta_max / T))
            else:
                accept = False
        
        if not accept:
            # Revert
            signs[idx] = old_sign
            for ap_id in index_to_aps[idx]:
                ap_sums[ap_id] -= delta
            # current_max stays the same
        else:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs[:]
        
        step += 1
    
    return best_signs
# EVOLVE-BLOCK-END
