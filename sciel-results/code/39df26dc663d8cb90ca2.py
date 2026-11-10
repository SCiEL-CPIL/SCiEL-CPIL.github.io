"""
Simulated annealing with multi-flip moves for minimizing max |sum| over APs.
Structural change: SA + multi-flip neighborhood + structured initialization.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1] with length >= 2.
    Returns list of index lists.
    """
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_indices.append(seq)
    return ap_indices

def _precompute_ap_containing(ap_indices, n):
    """
    For each index k, find which APs contain it.
    Returns list of lists of AP indices.
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _structured_init(n):
    """
    Initialize signs using a structured pattern based on quadratic residues mod 4
    or similar low-discrepancy construction.
    """
    signs = np.ones(n, dtype=np.int8)
    # Use a pattern: sign[i] = 1 if i mod 4 in {0, 1}, -1 if i mod 4 in {2, 3}
    # This gives balanced blocks and may help with AP sums
    for i in range(n):
        if i % 4 in (2, 3):
            signs[i] = -1
    return signs

def _compute_max_abs_sum(current_sums):
    """Compute max absolute sum from the AP sums array."""
    if len(current_sums) == 0:
        return 1
    return int(np.max(np.abs(current_sums)))

def _construct_signs_sa():
    """
    Construct signs using structured init + simulated annealing with multi-flip moves.
    """
    start_time = time.time()
    deadline = start_time + 58  # Safety margin (90s limit, leave buffer)
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Initialize with structured pattern
    signs = _structured_init(n)
    
    # Compute initial sums
    current_sums = np.zeros(num_aps, dtype=np.int32)
    for i in range(n):
        if len(ap_containing[i]) > 0:
            current_sums[ap_containing[i]] += signs[i]
    
    current_max = _compute_max_abs_sum(current_sums)
    
    # Track best
    best_signs = signs.copy()
    best_max = current_max
    
    # Simulated annealing parameters
    T0 = 10.0  # Initial temperature
    T_end = 0.1  # Final temperature
    cooling_rate = 0.995  # Geometric cooling
    
    rng = random.Random(12345)
    
    T = T0
    iteration = 0
    
    while time.time() < deadline and T > T_end:
        iteration += 1
        
        # Choose move type: 70% single flip, 30% multi-flip (2-3 positions)
        if rng.random() < 0.7:
            # Single flip
            pos = rng.randint(0, n - 1)
            flip_positions = [pos]
        else:
            # Multi-flip: 2 or 3 distinct positions
            num_flips = rng.choice([2, 3])
            flip_positions = rng.sample(range(n), num_flips)
        
        # Compute the change in sums for all affected APs
        # For each flipped position, the change in sign is -2 * old_sign
        # Collect all affected APs (union)
        affected_aps_set = set()
        for pos in flip_positions:
            for ap_idx in ap_containing[pos]:
                affected_aps_set.add(ap_idx)
        
        affected_aps = list(affected_aps_set)
        
        if len(affected_aps) == 0:
            continue
        
        # Save old values
        old_sums = current_sums[affected_aps].copy()
        old_signs = [signs[p] for p in flip_positions]
        
        # Apply flips
        for pos in flip_positions:
            old_s = signs[pos]
            signs[pos] = -old_s
            # Change in sum for each AP containing pos is -2 * old_s
            for ap_idx in ap_containing[pos]:
                current_sums[ap_idx] -= 2 * old_s
        
        # Compute new max
        new_max = _compute_max_abs_sum(current_sums)
        
        # Acceptance criterion (simulated annealing)
        delta = new_max - current_max
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept the move
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            # Reject: revert
            current_sums[affected_aps] = old_sums
            for idx, pos in enumerate(flip_positions):
                signs[pos] = old_signs[idx]
        
        # Cool down
        T *= cooling_rate
    
    # Final local search polish: single flips only, accept only improvements
    polish_deadline = deadline
    for _ in range(200):  # Limited polish iterations
        if time.time() >= polish_deadline:
            break
        pos = rng.randint(0, n - 1)
        old_sign = signs[pos]
        affected_aps = ap_containing[pos]
        if len(affected_aps) == 0:
            continue
        
        old_sums = current_sums[affected_aps].copy()
        signs[pos] = -old_sign
        current_sums[affected_aps] -= 2 * old_sign
        new_max = _compute_max_abs_sum(current_sums)
        
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            current_sums[affected_aps] = old_sums
            signs[pos] = old_sign
    
    return [int(x) for x in best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    signs = _construct_signs_sa()
    return signs

# EVOLVE-BLOCK-END
