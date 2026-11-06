"""Rewritten program: Max-SAT style local search for minimizing monochromatic 4-APs."""
import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0  # safety margin under 120s
    
    # Precompute all 4-term arithmetic progressions (0-indexed positions 0..255)
    # AP: (a, a+d, a+2d, a+3d) where a+3d < n, d >= 1
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for vectorized operations
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Helper: count monochromatic 4-APs for a given coloring
    def count_mono(color):
        # color is numpy array of 0/1, length n
        # For each AP, check if all four positions have the same color
        # Vectorized: get colors at the four positions
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        # Monochromatic if all equal: c0==c1==c2==c3
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    # Helper: compute which APs are monochromatic (boolean mask)
    def mono_mask(color):
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        return (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Helper: delta flip - how much does flipping position i change the mono count?
    # When we flip position i, only APs containing i can change status.
    # For each AP containing i, we check if it was mono before and after.
    def delta_flip(color, i):
        # Find all APs containing position i
        mask = (aps == i).any(axis=1)
        if not np.any(mask):
            return 0
        affected_aps = aps[mask]
        # Current mono status
        c0 = color[affected_aps[:, 0]]
        c1 = color[affected_aps[:, 1]]
        c2 = color[affected_aps[:, 2]]
        c3 = color[affected_aps[:, 3]]
        was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        # After flipping position i
        # For each affected AP, determine which position is i and flip it
        new_c0 = c0.copy()
        new_c1 = c1.copy()
        new_c2 = c2.copy()
        new_c3 = c3.copy()
        pos0 = (affected_aps[:, 0] == i)
        pos1 = (affected_aps[:, 1] == i)
        pos2 = (affected_aps[:, 2] == i)
        pos3 = (affected_aps[:, 3] == i)
        new_c0[pos0] = 1 - new_c0[pos0]
        new_c1[pos1] = 1 - new_c1[pos1]
        new_c2[pos2] = 1 - new_c2[pos2]
        new_c3[pos3] = 1 - new_c3[pos3]
        now_mono = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)
        # Delta = new_mono_count - old_mono_count
        return int(np.sum(now_mono) - np.sum(was_mono))
    
    # Local search: try flipping each position, accept if it reduces mono count
    def local_search(color, max_iters=5000):
        current_cost = count_mono(color)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try all positions, pick the best improvement
            best_delta = 0
            best_pos = -1
            for i in range(n):
                d = delta_flip(color, i)
                if d < best_delta:
                    best_delta = d
                    best_pos = i
            if best_pos >= 0 and best_delta < 0:
                color[best_pos] = 1 - color[best_pos]
                current_cost += best_delta
                improved = True
        return color, current_cost
    
    # Simulated annealing style local search with random restarts
    def sa_local_search(color, start_temp=10.0, end_temp=0.1, steps=20000):
        current_cost = count_mono(color)
        temp = start_temp
        for step in range(steps):
            if time.time() > deadline:
                break
            temp = start_temp * (end_temp / start_temp) ** (step / steps)
            i = random.randint(0, n - 1)
            d = delta_flip(color, i)
            if d <= 0 or random.random() < np.exp(-d / max(temp, 1e-6)):
                color[i] = 1 - color[i]
                current_cost += d
        return color, current_cost
    
    # Generate diverse initial solutions
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def random_coloring():
        return np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int8)
    
    def block_pattern(block_size):
        return np.array([(i // block_size) % 2 for i in range(n)], dtype=np.int8)
    
    def modular_pattern(mod):
        return np.array([i % mod % 2 for i in range(n)], dtype=np.int8)
    
    # List of initializations
    initializations = []
    initializations.append(thue_morse())
    initializations.append(block_pattern(2))
    initializations.append(block_pattern(3))
    initializations.append(block_pattern(4))
    initializations.append(block_pattern(5))
    initializations.append(block_pattern(8))
    initializations.append(modular_pattern(4))
    initializations.append(modular_pattern(8))
    initializations.append(modular_pattern(16))
    for _ in range(10):
        initializations.append(random_coloring())
    
    # Run local search on each initialization, keep best
    best_color = None
    best_cost = float('inf')
    
    for init_color in initializations:
        if time.time() > deadline - 5:
            break
        color = init_color.copy()
        # First do greedy local search
        color, cost = local_search(color, max_iters=2000)
        # Then do simulated annealing
        color, cost = sa_local_search(color, start_temp=5.0, end_temp=0.05, steps=10000)
        # Final greedy polish
        color, cost = local_search(color, max_iters=500)
        
        if cost < best_cost:
            best_cost = cost
            best_color = color.copy()
    
    # If we still have time, do more SA restarts from the best solution
    if best_color is not None and time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            color = best_color.copy()
            # Add some noise
            for _ in range(5):
                i = random.randint(0, n - 1)
                color[i] = 1 - color[i]
            color, cost = sa_local_search(color, start_temp=2.0, end_temp=0.01, steps=15000)
            color, cost = local_search(color, max_iters=300)
            if cost < best_cost:
                best_cost = cost
                best_color = color.copy()
    
    # Convert to list of ints
    return [int(x) for x in best_color]
