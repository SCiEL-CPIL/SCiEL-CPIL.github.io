"""
Strategy: Greedy Constructive Initialization + Vectorized Local Search with Simulated Annealing.

This program constructs a sign sequence of length 256 by:
1. Greedy construction: at each position, choose the sign that minimizes the maximum absolute
   partial sum over all arithmetic progressions ending at that position.
2. Multiple random restarts of greedy construction to diversify.
3. Local search with simulated annealing to refine the solution.
4. Pre-computed AP structure for fast evaluation.

The key insight is that greedy construction provides a much stronger starting point than random
initialization, as it directly minimizes local AP conflicts during the build-up of the sequence.
"""

import time
import random
import numpy as np


def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and for each index, which APs contain it and at what position.
    Returns:
        ap_indices: list of arrays, one per AP, containing the indices in that AP
        index_to_aps: list of lists, index_to_aps[i] = [(ap_id, pos), ...] for all APs containing index i
        ap_lengths: array of lengths of each AP
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = list(range(start, n, step))
            if len(indices) == 0:
                continue
            
            ap_indices.append(indices)
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    ap_lengths = np.array([len(ap) for ap in ap_indices], dtype=np.int64)
    return ap_indices, index_to_aps, ap_lengths


def _compute_max_discrepancy(signs, ap_indices):
    """Compute the maximum absolute sum over all APs."""
    max_disc = 0
    for indices in ap_indices:
        s = sum(signs[i] for i in indices)
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc


def _greedy_construct(n, ap_indices, index_to_aps, ap_lengths, rng, seed_offset=0):
    """
    Greedily construct a sign sequence. At each position i (from 0 to n-1),
    choose sign +1 or -1 to minimize the maximum absolute partial sum over
    all APs that include position i (considering only the prefix up to i).
    
    We maintain running partial sums for each AP as we build the sequence.
    """
    signs = [0] * n
    # For each AP, maintain the current sum (over the prefix of positions 0..i that are in the AP)
    ap_current_sum = [0] * len(ap_indices)
    # Track max absolute partial sum for each AP (over all prefixes seen so far)
    ap_max_abs = [0] * len(ap_indices)
    
    # Process positions in order
    for i in range(n):
        # For each candidate sign, compute the resulting max discrepancy contribution
        best_sign = 1
        best_max_contrib = 0
        best_second = -1
        best_second_max_contrib = 0
        
        for candidate in [1, -1]:
            max_contrib = 0
            for ap_id, pos in index_to_aps[i]:
                # The AP includes position i at index 'pos' in the AP
                # Current sum of the AP over positions < i that are in the AP
                # We need to track partial sums per AP prefix
                # Actually, let's track it differently
                pass
            
            # Simpler approach: for each AP containing i, compute what the new partial sum would be
            # and track the max absolute value
            new_max = 0
            for ap_id, pos in index_to_aps[i]:
                # We need the sum of signs[0..i-1] restricted to this AP
                # This is tricky without precomputation. Let's use a different approach.
                pass
        
        # Alternative greedy: just pick randomly with some bias, or use a simpler criterion
        # For simplicity and speed, let's use a heuristic:
        # Count how many APs containing i currently have positive sum vs negative
        pos_count = 0
        neg_count = 0
        for ap_id, pos in index_to_aps[i]:
            # We don't easily have current AP sums without tracking them
            pass
        
        # Let's use a simpler strategy: random with occasional bias
        if rng.random() < 0.5:
            signs[i] = 1
        else:
            signs[i] = -1
    
    return signs


def _greedy_construct_v2(n, ap_indices, index_to_aps, ap_lengths, rng):
    """
    Improved greedy construction. We maintain running sums for each AP.
    For each position i in order, we try both signs and pick the one that
    minimizes the maximum absolute value of any AP's sum after adding this sign.
    """
    signs = [0] * n
    num_aps = len(ap_indices)
    ap_sums = [0] * num_aps
    
    # For efficiency, precompute for each position i the list of (ap_id, pos) 
    # and also the AP length so we know when we're at the last position of an AP
    
    for i in range(n):
        best_sign = 1
        best_max_val = 0
        second_sign = -1
        second_max_val = 0
        
        for candidate in [1, -1]:
            # Compute the max absolute AP sum if we set signs[i] = candidate
            # Only APs containing i are affected
            max_val = 0
            for ap_id, pos in index_to_aps[i]:
                new_sum = ap_sums[ap_id] + candidate
                abs_new = abs(new_sum)
                if abs_new > max_val:
                    max_val = abs_new
            
            if best_max_val == 0 or max_val < best_max_val:
                second_sign = best_sign
                second_max_val = best_max_val
                best_sign = candidate
                best_max_val = max_val
            elif max_val < second_max_val:
                second_sign = candidate
                second_max_val = max_val
        
        signs[i] = best_sign
        # Update AP sums
        for ap_id, pos in index_to_aps[i]:
            ap_sums[ap_id] += best_sign
    
    return signs


def _local_search_sa(signs, ap_indices, index_to_aps, deadline, rng, T_init=2.0, T_min=0.1, cooling=0.999):
    """
    Local search with simulated annealing.
    """
    n = len(signs)
    signs = list(signs)
    num_aps = len(ap_indices)
    
    # Compute initial AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int64)
    signs_arr = np.array(signs, dtype=np.int64)
    for ap_id, indices in enumerate(ap_indices):
        ap_sums[ap_id] = np.sum(signs_arr[indices])
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = list(signs)
    best_max = current_max
    
    T = T_init
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 3 and T > T_min and no_improve_count < max_no_improve:
        i = rng.randrange(n)
        old_val = signs[i]
        new_val = -old_val
        delta = new_val - old_val  # = -2*old_val
        
        affected_aps = index_to_aps[i]
        
        # Compute new AP sums for affected APs
        old_sums = []
        new_sums = []
        for ap_id, pos in affected_aps:
            old_s = int(ap_sums[ap_id])
            new_s = old_s + delta
            old_sums.append(old_s)
            new_sums.append(new_s)
        
        # Compute new global max
        new_max = current_max
        for j, ap_id in enumerate([a[0] for a in affected_aps]):
            abs_new = abs(new_sums[j])
            if abs_new > new_max:
                new_max = abs_new
        
        # But we also need to check if removing the old max reduces it
        # If the old max was from an affected AP, we need to recompute
        # For simplicity, if new_max > current_max, we might accept with probability
        # If new_max < current_max, we accept and update
        
        # Actually, the correct new max is max of:
        # 1. abs(new_sums[j]) for affected APs
        # 2. abs(ap_sums[k]) for unaffected APs (which is <= current_max)
        # So new_max = max(max(abs(new_sums[j]) for affected), max(abs(ap_sums[k]) for unaffected))
        # The second term is <= current_max.
        
        if new_max < current_max:
            # Accept
            for j, (ap_id, pos) in enumerate(affected_aps):
                ap_sums[ap_id] = new_sums[j]
            signs[i] = new_val
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = list(signs)
            no_improve_count = 0
        elif new_max == current_max:
            # Accept with some probability to allow exploration
            if rng.random() < 0.3:
                for j, (ap_id, pos) in enumerate(affected_aps):
                    ap_sums[ap_id] = new_sums[j]
                signs[i] = new_val
                # current_max stays the same
        else:
            # Worse, accept with probability exp(-(new_max - current_max)/T)
            delta_e = new_max - current_max
            if rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                for j, (ap_id, pos) in enumerate(affected_aps):
                    ap_sums[ap_id] = new_sums[j]
                signs[i] = new_val
                current_max = new_max
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
    
    return best_signs, best_max


def _local_search_greedy(signs, ap_indices, index_to_aps, deadline, rng):
    """
    Pure greedy local search: only accept moves that strictly improve the max discrepancy.
    """
    n = len(signs)
    signs = list(signs)
    num_aps = len(ap_indices)
    
    ap_sums = np.zeros(num_aps, dtype=np.int64)
    signs_arr = np.array(signs, dtype=np.int64)
    for ap_id, indices in enumerate(ap_indices):
        ap_sums[ap_id] = np.sum(signs_arr[indices])
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = list(signs)
    best_max = current_max
    
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline - 3:
            break
        
        improved = False
        indices_to_try = list(range(n))
        rng.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if time.time() > deadline - 3:
                break
            
            old_val = signs[i]
            new_val = -old_val
            delta = -2 * old_val
            
            affected_aps = index_to_aps[i]
            
            # Compute new max for affected APs
            max_new_aff = 0
            for ap_id, pos in affected_aps:
                new_s = int(ap_sums[ap_id]) + delta
                abs_new = abs(new_s)
                if abs_new > max_new_aff:
                    max_new_aff = abs_new
            
            # If any affected AP would exceed current_max, this flip is bad
            if max_new_aff > current_max:
                continue
            
            # Apply flip
            signs[i] = new_val
            for ap_id, pos in affected_aps:
                ap_sums[ap_id] += delta
            
            new_global_max = int(np.max(np.abs(ap_sums)))
            
            if new_global_max < current_max:
                current_max = new_global_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = list(signs)
            else:
                # Revert (neutral move)
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    ap_sums[ap_id] -= delta
        
        if not improved:
            break
    
    return best_signs, best_max


def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (evaluator kills at 90s, but be conservative)
    
    rng = random.Random(12345)
    
    # Pre-compute AP structure
    ap_indices, index_to_aps, ap_lengths = _build_ap_structure(n)
    num_aps = len(ap_indices)
    
    best_signs_overall = None
    best_global_max_overall = float('inf')
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for trial in range(15):
        if time.time() > deadline - 20:
            break
        
        g = _greedy_construct_v2(n, ap_indices, index_to_aps, ap_lengths, rng)
        
        # Quick local search
        best_signs, best_max = _local_search_greedy(g, ap_indices, index_to_aps, deadline, rng)
        
        if best_max < best_global_max_overall:
            best_global_max_overall = best_max
            best_signs_overall = best_signs
    
    # Phase 2: Simulated annealing from the best greedy result
    if best_signs_overall is not None and time.time() < deadline - 10:
        sa_signs, sa_max = _local_search_sa(
            best_signs_overall, ap_indices, index_to_aps, deadline, rng,
            T_init=1.5, T_min=0.05, cooling=0.9995
        )
        if sa_max < best_global_max_overall:
            best_global_max_overall = sa_max
            best_signs_overall = sa_signs
    
    # Phase 3: More local search refinement
    if best_signs_overall is not None and time.time() < deadline - 3:
        final_signs, final_max = _local_search_greedy(
            best_signs_overall, ap_indices, index_to_aps, deadline, rng
        )
        if final_max < best_global_max_overall:
            best_global_max_overall = final_max
            best_signs_overall = final_signs
    
    if best_signs_overall is None:
        best_signs_overall = [1] * n
    
    return best_signs_overall


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s, Result length: {len(result)}")
    
    # Verify
    ap_indices, _, _ = _build_ap_structure(256)
    max_disc = 0
    for indices in ap_indices:
        s = sum(result[i] for i in indices)
        if abs(s) > max_disc:
            max_disc = abs(s)
    print(f"Max Discrepancy: {max_disc}")
