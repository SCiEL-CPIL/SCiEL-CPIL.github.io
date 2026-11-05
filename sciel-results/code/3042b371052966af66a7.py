"""
Optimized two-colouring of [1,256] minimizing monochromatic 4-term APs.
Uses vectorized delta computation with NumPy for speed, enabling many more
local search sweeps and restarts within the time budget.
"""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    aps_list = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps_list.append((i0, i1, i2, i3))
    
    num_aps = len(aps_list)
    aps_arr = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, precompute which APs it's in and the other 3 positions
    # pos_to_aps[pos] = list of (ap_index, [other_pos1, other_pos2, other_pos3])
    pos_to_aps = [[] for _ in range(256)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps_list):
        for pos in (i0, i1, i2, i3):
            others = [x for x in (i0, i1, i2, i3) if x != pos]
            pos_to_aps[pos].append((ap_idx, others))
    
    # Convert to numpy arrays for vectorized delta computation
    # For each position, store:
    #   ap_indices: array of AP indices
    #   other_positions: array of shape (n_aps_for_pos, 3)
    pos_ap_indices = []
    pos_other_positions = []
    for pos in range(256):
        if pos_to_aps[pos]:
            indices = [x[0] for x in pos_to_aps[pos]]
            others = [x[1] for x in pos_to_aps[pos]]
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_other_positions.append(np.array(others, dtype=np.int32))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_other_positions.append(np.array([], dtype=np.int32))
    
    # Count monochromatic 4-APs (vectorized)
    def count_mono(colors):
        # colors is a numpy array of 0/1
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    # Vectorized delta computation for flipping position pos
    def calc_delta_vec(colors, pos):
        """Returns delta in mono count when flipping position pos."""
        idx = pos_ap_indices[pos]
        others = pos_other_positions[pos]
        if len(idx) == 0:
            return 0
        
        # Get colors of the other 3 positions in each AP
        o0 = colors[others[:, 0]]
        o1 = colors[others[:, 1]]
        o2 = colors[others[:, 2]]
        
        # Check if all 4 are currently monochromatic
        cur_color = colors[pos]
        all_same_now = (o0 == o1) & (o1 == o2) & (o2 == cur_color)
        
        # Check if all 4 would be monochromatic after flip
        new_color = 1 - cur_color
        all_same_after = (o0 == o1) & (o1 == o2) & (o2 == new_color)
        
        # Delta = (# added) - (# removed)
        removed = np.sum(all_same_now & ~all_same_after)
        added = np.sum(~all_same_now & all_same_after)
        return int(added - removed)
    
    # Greedy local search: repeatedly flip positions that reduce count
    def greedy_local_search(colors, current_count, max_sweeps=50):
        improved = True
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline:
            improved = False
            sweep += 1
            for pos in range(256):
                delta = calc_delta_vec(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    improved = True
        return current_count
    
    # Best result found so far
    best_colors = None
    best_count = float('inf')
    
    rng = random.Random(12345)
    np_rng = np.random.RandomState(67890)
    
    # Multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Choose initialization strategy
        if num_restarts == 1:
            # Start with alternating pattern
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif num_restarts % 3 == 0:
            # Start with block pattern
            colors = np.array([((i // 4) % 2) for i in range(256)], dtype=np.int8)
        elif num_restarts % 3 == 1:
            # Start with random
            colors = np_rng.randint(0, 2, size=256).astype(np.int8)
        else:
            # Start with a mix: first half alternating, second half random
            colors = np.zeros(256, dtype=np.int8)
            colors[:128] = np.array([(i % 2) for i in range(128)], dtype=np.int8)
            colors[128:] = np_rng.randint(0, 2, size=128).astype(np.int8)
        
        current_count = count_mono(colors)
        
        # Greedy local search
        current_count = greedy_local_search(colors, current_count, max_sweeps=100)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # If we haven't found a good solution yet, do simulated annealing
        if best_count > 30 and time.time() < deadline - 5:
            # Simulated annealing phase
            T = 20.0
            T_min = 0.5
            cooling = 0.999
            sa_steps = 0
            max_sa_steps = 50000
            
            while T > T_min and sa_steps < max_sa_steps and time.time() < deadline - 5:
                pos = rng.randint(0, 255)
                delta = calc_delta_vec(colors, pos)
                
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                
                T *= cooling
                sa_steps += 2  # Count both accepted and rejected as steps for budget
                
                # Periodic greedy cleanup
                if sa_steps % 5000 == 0 and time.time() < deadline - 5:
                    current_count = greedy_local_search(colors, current_count, max_sweeps=10)
        
        # Additional targeted improvement: try all single flips
        if time.time() < deadline - 2:
            for pos in range(256):
                delta = calc_delta_vec(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors.copy()
    
    # Final exhaustive single-flip check on best solution
    if best_colors is not None and time.time() < deadline - 1:
        final = best_colors.copy()
        final_count = best_count
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for pos in range(256):
                delta = calc_delta_vec(final, pos)
                if delta < 0:
                    final[pos] = 1 - final[pos]
                    final_count += delta
                    changed = True
        if final_count < best_count:
            best_count = final_count
            best_colors = final
    
    return [int(c) for c in best_colors]
