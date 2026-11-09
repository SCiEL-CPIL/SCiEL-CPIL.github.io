"""Optimized program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_aps(n):
    """Precompute all arithmetic progressions and their index lists."""
    aps = []
    for d in range(1, n + 1):
        for start in range(0, n, d):
            idxs = list(range(start, n, d))
            if idxs:
                aps.append(idxs)
    return aps

def _compute_max_discrepancy(signs, aps):
    """Compute the maximum absolute signed sum over all APs."""
    max_disc = 0
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def _compute_discrepancy_array(signs, aps):
    """Compute discrepancy for all APs, return array of absolute sums."""
    n_aps = len(aps)
    disc = np.zeros(n_aps, dtype=np.int32)
    for i, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += signs[idx]
        disc[i] = abs(s)
    return disc

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill
    
    # Precompute all APs
    aps = _precompute_aps(n)
    n_aps = len(aps)
    
    # For each index, precompute which APs contain it
    index_to_aps = [[] for _ in range(n)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(ap_id)
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Compute initial discrepancies
    disc = _compute_discrepancy_array(signs, aps)
    current_max = int(disc.max())
    
    # Track best solution
    best_signs = signs.copy()
    best_max = current_max
    
    # Simulated annealing / local search
    # We'll use a simple hill-climbing with occasional random restarts
    
    temperature = 10.0
    cooling = 0.9995
    min_temperature = 0.1
    
    step = 0
    while time.time() - start_time < time_limit:
        # Check if we should stop
        if temperature < min_temperature:
            # Try a few more steps then break
            if step > 100000:
                break
        
        # Choose a random index to flip
        idx = rng.randrange(n)
        
        # Compute the change in discrepancies if we flip signs[idx]
        # For each AP containing idx, the sum changes by -2*signs[idx]
        delta = -2 * signs[idx]
        
        # Find the AP with the maximum absolute sum that contains idx
        # We want to know how flipping affects the max discrepancy
        
        # Quick check: compute new max discrepancy approximately
        # For each AP containing idx, new_abs = abs(disc[ap_id] + delta * (1 if signs[idx] > 0 else -1))
        # Wait, let's think more carefully.
        
        # Current sum for AP containing idx: let's call it S
        # After flip: S + delta where delta = -2 * signs[idx]
        # New absolute sum: abs(S + delta)
        # We don't store S, we store abs(S) = disc[ap_id]
        # We need to know the sign of S to compute the new abs value
        
        # Let's store signed sums instead
        # Actually, let's just compute the new max by checking all APs containing idx
        
        affected_aps = index_to_aps[idx]
        
        # Compute what the new discrepancies would be for affected APs
        # We need the actual signed sums, not just absolute values
        # Let's recompute signed sums for affected APs
        
        max_new_disc = current_max  # Start with current max (for unaffected APs)
        
        # For each affected AP, compute new signed sum
        for ap_id in affected_aps:
            ap = aps[ap_id]
            # Current signed sum
            s = 0
            for i in ap:
                s += signs[i]
            # After flipping idx:
            s_new = s + delta
            abs_new = abs(s_new)
            if abs_new > max_new_disc:
                max_new_disc = abs_new
        
        # Decide whether to accept the flip
        if max_new_disc <= current_max:
            # Accept (improvement or equal)
            signs[idx] = -signs[idx]
            current_max = max_new_disc
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Metropolis acceptance
            delta_disc = max_new_disc - current_max
            if rng.random() < np.exp(-delta_disc / temperature):
                signs[idx] = -signs[idx]
                current_max = max_new_disc
        
        # Cool down
        temperature *= cooling
        step += 1
        
        # Periodic check for time
        if step % 1000 == 0:
            if time.time() - start_time >= time_limit - 5:
                break
    
    # Return the best solution found
    return list(best_signs)
# EVOLVE-BLOCK-END
