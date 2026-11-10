import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # Precompute for each position, the indices of the other 3 elements in every AP containing it.
    # pos_ap_others[p] is a numpy array of shape (K, 3), where K is the number of APs containing p.
    pos_ap_others = [None] * N
    for p in range(N):
        others_list = []
        for i in range(total_aps):
            row = aps[i]
            # Check if p is in this AP
            if p in row:
                # Find the index of p in the row to exclude it
                idx = np.where(row == p)[0][0]
                others = [row[j] for j in range(4) if j != idx]
                others_list.append(others)
        if others_list:
            pos_ap_others[p] = np.array(others_list, dtype=np.int32)
        else:
            pos_ap_others[p] = np.empty((0, 3), dtype=np.int32)

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy. colors is numpy array of 0/1."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        """Compute change in mono count if we flip position p using precomputed others."""
        old_color = colors[p]
        others = pos_ap_others[p]
        n_aps = others.shape[0]
        if n_aps == 0:
            return 0
        
        # Get colors of the other 3 positions in each AP
        # others has shape (n_aps, 3)
        other_colors = colors[others] # shape (n_aps, 3)
        
        # Check if the 3 other positions are all the same color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])
        
        # Before flip: AP is mono if all 3 others are same AND that color is old_color
        mono_before = all_same & (other_colors[:, 0] == old_color)
        
        # After flip: AP is mono if all 3 others are same AND that color is new_color (1 - old_color)
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)
        
        return int(mono_after.sum()) - int(mono_before.sum())

    best_colors = None
    best_count = float('inf')
    
    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Strategy: multiple restarts with simulated annealing + local search
    num_restarts = 20  # Increased restarts due to faster delta calculation
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            # Alternating pairs (seed pattern)
            colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 1:
            # Random
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        elif restart == 2:
            # Thicker blocks: 000111000111...
            colors = np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 3:
            # Thicker blocks: 0000111100001111...
            colors = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 4:
            # Thicker blocks: 00000111110000011111...
            colors = np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 5:
            # Thicker blocks: 000000111111000000111111...
            colors = np.array([(i // 6) % 2 for i in range(N)], dtype=np.int8)
        else:
            if best_colors is not None:
                colors = best_colors.copy()
                # Perturb ~15% of bits
                n_flips = max(5, N // 7)
                flip_positions = np.random.choice(N, size=n_flips, replace=False)
                colors[flip_positions] ^= 1
            else:
                colors = np.random.randint(0, 2, size=N).astype(np.int8)
        
        current_count = count_mono(colors)
        update_best(colors, current_count)
        
        # Simulated Annealing
        T = 3.0
        T_min = 0.05
        cooling = 0.9998
        max_sa_steps = 100000  # High limit, controlled by time
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random position
            p = np.random.randint(0, N)
            
            # Compute delta
            delta = fast_delta(colors, p)
            
            if delta <= 0:
                colors[p] ^= 1
                current_count += delta
                update_best(colors, current_count)
            else:
                # Accept with probability exp(-delta/T)
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        colors[p] ^= 1
                        current_count += delta
                # else reject (T too low)
            
            T = max(T_min, T * cooling)
        
        # Local search: hill climbing
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
        
        # Second pass of local search (might find more improvements)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for p in range(N):
                if time.time() > deadline - 2:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
