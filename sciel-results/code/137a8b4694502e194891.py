"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    # For each AP, store the indices (0-based)
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            # AP: start, start+d, start+2d, ...
            # Length = floor((N-1 - start) / d) + 1
            # Only include if length >= 2
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Convert APs to list of numpy arrays for faster indexing
    aps_np = [np.array(ap, dtype=np.int32) for ap in aps]
    
    # For each index, precompute which APs it belongs to
    # This allows us to quickly update affected APs when flipping a sign
    index_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps_np):
        for idx in ap:
            index_to_aps[idx].append(i)
    
    # Precompute AP sums for current signs
    def compute_all_ap_sums(signs):
        """Compute sums for all APs."""
        sums = np.zeros(num_aps, dtype=np.int32)
        for i, ap in enumerate(aps_np):
            sums[i] = np.sum(signs[ap])
        return sums
    
    def compute_max_abs_sum_from_sums(sums):
        """Compute maximum absolute sum from precomputed AP sums."""
        if len(sums) == 0:
            return 1
        return max(1, int(np.max(np.abs(sums))))
    
    # Simulated Annealing with incremental updates
    rng = random.Random(42)
    
    # Initial solution: random signs
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    current_sums = compute_all_ap_sums(signs)
    current_cost = compute_max_abs_sum_from_sums(current_sums)
    best_signs = signs.copy()
    best_sums = current_sums.copy()
    best_cost = current_cost
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    
    steps = 0
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a random index to flip
        idx = rng.randrange(N)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        # Update affected APs incrementally
        affected_aps = index_to_aps[idx]
        if affected_aps:
            for ap_idx in affected_aps:
                current_sums[ap_idx] += 2 * old_sign  # Flipping changes sum by -2*old_sign = 2*new_sign... wait
                # Actually: old contribution was old_sign, new contribution is -old_sign
                # Change in sum = new_sign - old_sign = -old_sign - old_sign = -2*old_sign
                # So: new_sum = old_sum + (-2*old_sign)
        
        new_cost = compute_max_abs_sum_from_sums(current_sums)
        
        if new_cost <= current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
                best_sums = current_sums.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    signs[idx] = old_sign
                    for ap_idx in affected_aps:
                        current_sums[ap_idx] -= 2 * old_sign  # Revert the change
        
        T *= cooling_rate
        steps += 1
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
