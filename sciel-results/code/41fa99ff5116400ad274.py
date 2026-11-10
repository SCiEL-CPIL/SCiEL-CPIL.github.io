"""
Rewritten program: Local search with fast AP sum updates for minimizing max |sum| over APs.
Uses quadratic character initialization and efficient local search.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1].
    Returns:
        ap_indices: List of lists, where ap_indices[j] is a list of indices for the j-th AP.
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
    Returns:
        ap_containing: List of lists, where ap_containing[k] is a list of AP indices j 
                       such that k is in ap_indices[j].
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _quadratic_character_init(n, p=257):
    """
    Initialize signs using quadratic character modulo prime p.
    For prime p, the Legendre symbol (a/p) is 1 if a is a quadratic residue mod p, -1 otherwise.
    """
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Use i+1 mod p for the quadratic character
        a = (i + 1) % p
        # Compute Legendre symbol using Euler's criterion
        if a == 0:
            signs[i] = 1
        else:
            # (a/p) = a^((p-1)/2) mod p
            if pow(a, (p - 1) // 2, p) == 1:
                signs[i] = 1
            else:
                signs[i] = -1
    return signs

def _construct_signs_fast():
    """
    Construct signs using multi-start local search with quadratic character init.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    global_best_signs = None
    global_best_max = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline - 2: # Leave 2s for final checks
        restart_count += 1
        # Initialize using quadratic character
        signs = _quadratic_character_init(n)
        
        # Compute initial sums
        current_sums = np.zeros(num_aps, dtype=np.int32)
        for j, indices in enumerate(ap_indices):
            current_sums[j] = np.sum(signs[indices])
        
        current_max = int(np.max(np.abs(current_sums)))
        
        if current_max < global_best_max:
            global_best_max = current_max
            global_best_signs = signs.copy()
        
        # Local Search: Hill Climbing with random restarts on stagnation
        rng = random.Random(1000 + restart_count)
        no_improve_count = 0
        max_no_improve = 100 # Lower threshold to trigger restarts more often
        
        while time.time() < deadline - 1:
            pos = rng.randint(0, n - 1)
            
            old_sign = signs[pos]
            new_sign = -old_sign
            
            affected_aps = ap_containing[pos]
            if len(affected_aps) == 0:
                continue
                
            # Save old sums for affected APs
            old_sums = current_sums[affected_aps].copy()
            
            # Apply flip: change is -2 * old_sign
            delta = -2 * old_sign
            current_sums[affected_aps] += delta
            signs[pos] = new_sign
            
            # Check new max efficiently
            new_affected_max = int(np.max(np.abs(current_sums[affected_aps])))
            
            # Determine new global max for this solution
            # If new_affected_max > current_max, new_max = new_affected_max
            # Else new_max = current_max (since other sums didn't change)
            if new_affected_max > current_max:
                new_max = new_affected_max
            else:
                new_max = current_max
            
            # Acceptance criteria:
            # 1. If new_max < current_max: Accept (Improvement)
            # 2. If new_max == current_max: Accept with prob 0.5 (Plateau exploration)
            # 3. If new_max > current_max: Reject
            
            if new_max < current_max:
                current_max = new_max
                no_improve_count = 0
                # Update global best if this is better
                if current_max < global_best_max:
                    global_best_max = current_max
                    global_best_signs = signs.copy()
            elif new_max == current_max:
                if rng.random() < 0.5:
                    no_improve_count += 1
                else:
                    # Revert
                    current_sums[affected_aps] = old_sums
                    signs[pos] = old_sign
                    no_improve_count += 1
            else:
                # Revert
                current_sums[affected_aps] = old_sums
                signs[pos] = old_sign
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
        
    if global_best_signs is None:
        return [1] * n
        
    return [int(x) for x in global_best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    # Use the fast construction method
    signs = _construct_signs_fast()
    return signs

# EVOLVE-BLOCK-END
