"""
Simulated Annealing approach for minimizing max |sum| over APs.
Structural change: Replaces greedy+hill-climbing with Simulated Annealing
to escape local minima.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1] with length > 1.
    Returns list of lists of indices.
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
    Returns list of lists where ap_containing[k] is list of AP indices j containing k.
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _construct_signs_sa():
    """
    Construct signs using Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Precompute affected APs as numpy arrays for faster indexing
    affected_aps_list = [np.array(aps, dtype=np.int32) if len(aps) > 0 else np.array([], dtype=np.int32) 
                         for aps in ap_containing]
    
    # Initialize signs randomly
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Initialize AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for j, indices in enumerate(ap_indices):
        # sum signs at indices
        ap_sums[j] = np.sum(signs[indices])
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    # We don't know the number of steps in advance, so we use a time-based cooling
    # Or we can estimate steps. Let's do time-based.
    # Alternatively, we can just run until deadline and cool based on time elapsed.
    
    while time.time() < deadline:
        # Cooling: exponential decay based on time
        elapsed = time.time() - start_time
        # Map elapsed time to temperature. 
        # Let's assume we want to reach T_end at deadline.
        # T = T_start * exp(-k * elapsed)
        # T_end = T_start * exp(-k * (deadline-start))
        # k = ln(T_start/T_end) / (deadline-start)
        # We can precompute k
        total_time = deadline - start_time
        k = np.log(T_start / T_end) / total_time
        T = T_start * np.exp(-k * elapsed)
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        affected = affected_aps_list[pos]
        
        if len(affected) == 0:
            continue
            
        old_sign = signs[pos]
        
        # Apply flip
        # The change in sum for each affected AP is -2 * old_sign
        delta = -2 * old_sign
        old_sums_copy = ap_sums[affected].copy()
        ap_sums[affected] += delta
        
        # Calculate new max
        # Optimization: 
        # If the current global max was in 'affected', we must recompute global max.
        # Otherwise, new_max = max(old_max, max(abs(ap_sums[affected])))
        
        # To keep it simple and robust for N=256 (num_aps ~ 1500), 
        # np.max is very fast. But we can try the optimization.
        
        # Check if any AP in 'affected' had the old max
        # This check is O(len(affected)).
        # If not, we can just compare.
        
        # For simplicity and speed in Python, let's just compute the max of affected 
        # and see if it's bigger than current_max. 
        # If current_max was in affected, we lost track of the second max.
        # So if old_max was in affected, we MUST recompute global max.
        
        # How to check if old_max was in affected?
        # We can check if any abs(ap_sums[affected]) == old_max before the flip? 
        # No, because sums changed.
        # We can check if any abs(old_sums_copy) == current_max.
        
        had_max_in_affected = np.any(np.abs(old_sums_copy) == current_max)
        
        if had_max_in_affected:
            new_max = int(np.max(np.abs(ap_sums)))
        else:
            new_affected_max = int(np.max(np.abs(ap_sums[affected]))) if len(affected) > 0 else 0
            new_max = max(current_max, new_affected_max)
        
        # Acceptance Criterion
        if new_max < current_max:
            # Always accept improvement
            current_max = new_max
            signs[pos] = -old_sign
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Delta is new_max - current_max (>= 0)
            delta_e = new_max - current_max
            if delta_e == 0:
                # Accept with some probability to explore? 
                # Usually in SA, we accept if delta < 0. If delta == 0, accept with prob 1 or 0.5?
                # Let's accept to encourage movement.
                signs[pos] = -old_sign
                current_max = new_max # unchanged
            else:
                # Accept with probability exp(-delta_e / T)
                if T > 0:
                    prob = np.exp(-delta_e / T)
                    if rng.random() < prob:
                        signs[pos] = -old_sign
                        current_max = new_max
                    else:
                        # Revert
                        ap_sums[affected] = old_sums_copy
                else:
                    # Revert
                    ap_sums[affected] = old_sums_copy

    return [int(x) for x in best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    signs = _construct_signs_sa()
    return signs

# EVOLVE-BLOCK-END
