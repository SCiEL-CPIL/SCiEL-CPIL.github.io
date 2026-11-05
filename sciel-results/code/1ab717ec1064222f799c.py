import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 145
    
    # Precompute all 4-APs
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))
    
    aps_np = np.array(aps_list, dtype=np.int32)
    M = len(aps_np)
    
    # Precompute for each position, the indices of APs it belongs to
    # and the indices of the other 3 elements in those APs
    pos_aps_indices = [[] for _ in range(N)]
    pos_aps_others = [[] for _ in range(N)]
    
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps_indices[a].append(idx)
        pos_aps_others[a].append((b, c, d_pos))
        
        pos_aps_indices[b].append(idx)
        pos_aps_others[b].append((a, c, d_pos))
        
        pos_aps_indices[c].append(idx)
        pos_aps_others[c].append((a, b, d_pos))
        
        pos_aps_indices[d_pos].append(idx)
        pos_aps_others[d_pos].append((a, b, c))

    def count_mono(cols_np):
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def delta_1_flip(cols_np, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols_np[i]
        total_delta = 0
        for ap_idx, (o1, o2, o3) in zip(pos_aps_indices[i], pos_aps_others[i]):
            # Check if o1, o2, o3 are all same color
            if cols_np[o1] == cols_np[o2] == cols_np[o3]:
                if cols_np[o1] == c_i:
                    # Currently mono, flipping i breaks it
                    total_delta -= 1
                else:
                    # Currently not mono (3 same, 1 diff), flipping i makes it mono
                    total_delta += 1
        return total_delta

    def delta_2_flip(cols_np, i, j):
        """Calculate cost delta if we flip positions i and j simultaneously."""
        c_i = cols_np[i]
        c_j = cols_np[j]
        
        # We need to consider all APs that contain i OR j
        # If an AP contains both i and j, we must handle it carefully to avoid double counting or incorrect logic
        # If an AP contains only i, the other 3 are fixed.
        # If an AP contains only j, the other 3 are fixed.
        
        # To be safe and simple, let's just recalculate the status of affected APs.
        # Affected APs are those in pos_aps_indices[i] or pos_aps_indices[j].
        
        # Create a set of AP indices to check
        check_aps = set(pos_aps_indices[i])
        check_aps.update(pos_aps_indices[j])
        
        delta = 0
        for ap_idx in check_aps:
            a, b, c, d_pos = aps_np[ap_idx]
            
            # Get current colors
            cols_before = [cols_np[a], cols_np[b], cols_np[c], cols_np[d_pos]]
            
            # Apply flip to i and j
            cols_after = cols_before[:]
            if a == i: cols_after[0] = 1 - cols_after[0]
            if b == i: cols_after[1] = 1 - cols_after[1]
            if c == i: cols_after[2] = 1 - cols_after[2]
            if d_pos == i: cols_after[3] = 1 - cols_after[3]
            
            if a == j: cols_after[0] = 1 - cols_after[0]
            if b == j: cols_after[1] = 1 - cols_after[1]
            if c == j: cols_after[2] = 1 - cols_after[2]
            if d_pos == j: cols_after[3] = 1 - cols_after[3]
            
            mono_before = (cols_before[0] == cols_before[1] == cols_before[2] == cols_before[3])
            mono_after = (cols_after[0] == cols_after[1] == cols_after[2] == cols_after[3])
            
            if mono_before and not mono_after:
                delta -= 1
            elif not mono_before and mono_after:
                delta += 1
            # else: 0 change
            
        return delta

    def local_search_1_flip(cols, current_cost):
        """Greedy 1-flip local search."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_1_flip(cols_np, i)
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
        return cols, current_cost

    def local_search_2_flip(cols, current_cost):
        """Greedy 2-flip local search."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = delta_2_flip(cols_np, i, j)
                    if d < 0:
                        cols[i] = 1 - cols[i]
                        cols[j] = 1 - cols[j]
                        cols_np[i] = 1 - cols_np[i]
                        cols_np[j] = 1 - cols_np[j]
                        current_cost += d
                        improved = True
        return cols, current_cost

    def generate_initial():
        patterns = []
        # Thue-Morse
        tm = [bin(i+1).count('1') % 2 for i in range(N)]
        patterns.append(tm)
        patterns.append([1-c for c in tm])
        
        # Periodic
        for p in [2, 3, 4, 5, 7, 8]:
            patterns.append([i % p % 2 for i in range(N)])
            
        # Random
        rng = random.Random(42)
        for _ in range(5):
            patterns.append([rng.randint(0, 1) for _ in range(N)])
            
        # Score and sort
        scored = []
        for p in patterns:
            c = count_mono(np.array(p, dtype=np.int8))
            scored.append((c, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored]

    initial_patterns = generate_initial()
    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))
    
    # ILS Loop
    rng = random.Random(123)
    
    while time.time() < deadline - 5:
        # Perturb the current best or a random initial
        if rng.random() < 0.7:
            current = global_best_colors[:]
        else:
            current = initial_patterns[rng.randint(0, len(initial_patterns)-1)][:]
            
        # Perturbation: Flip a random block or random set
        if rng.random() < 0.5:
            # Block flip
            start = rng.randint(0, N - 10)
            length = rng.randint(3, 10)
            for k in range(length):
                current[start + k] = 1 - current[start + k]
        else:
            # Random set flip
            num_flips = rng.randint(3, 8)
            indices = rng.sample(range(N), num_flips)
            for idx in indices:
                current[idx] = 1 - current[idx]
                
        # Calculate cost
        current_cost = count_mono(np.array(current, dtype=np.int8))
        
        # Local Search Phase 1: 1-flip
        current, current_cost = local_search_1_flip(current, current_cost)
        
        # Check time
        if time.time() > deadline - 10:
            if current_cost < global_best_cost:
                global_best_cost = current_cost
                global_best_colors = current[:]
            continue
            
        # Local Search Phase 2: 2-flip
        # This is slower, so we might limit it or check time more frequently
        # For N=256, 2-flip search is O(N^2 * K). 
        # Let's try it. If it's too slow, we can skip or limit iterations.
        
        # Optimization: Only do 2-flip if 1-flip didn't improve much? 
        # No, always do it to escape 1-flip minima.
        
        current, current_cost = local_search_2_flip(current, current_cost)
        
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_colors = current[:]

    return global_best_colors
