import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 145
    
    # Precompute all 4-term arithmetic progressions in [0, 255] (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1 and a+3d < 256
    aps_list = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            aps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_np = np.array(aps_list, dtype=np.int32)
    M = len(aps_np)
    
    # For each position, precompute which APs it belongs to and the other 3 positions
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))
    
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))
    
    def count_mono(cols_np):
        """Vectorized count of monochromatic 4-APs."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def all_deltas(cols_np):
        """Compute delta for flipping each position. Returns array of shape (N,)."""
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            arr = pos_aps_arr[i]
            if len(arr) == 0:
                continue
            c_i = cols_np[i]
            others = arr[:, 1:]  # shape (k, 3)
            c_others = cols_np[others]  # shape (k, 3)
            all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
            if not np.any(all_same):
                continue
            matching = int(np.sum(c_others[all_same, 0] == c_i))
            n_same = int(np.sum(all_same))
            deltas[i] = -(matching) + (n_same - matching)
        return deltas
    
    def batch_polish(cols, current_cost, time_limit):
        """Batch best-improvement polish using vectorized deltas."""
        cols_np = np.array(cols, dtype=np.int8)
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            deltas = all_deltas(cols_np)
            neg_mask = deltas < 0
            if np.any(neg_mask):
                for i in np.where(neg_mask)[0]:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                current_cost += int(np.sum(deltas[neg_mask]))
                improved = True
        return cols, current_cost
    
    def greedy_polish(cols, current_cost, max_sweeps=50):
        """Sequential best-improvement polish."""
        cols_np = np.array(cols, dtype=np.int8)
        for sweep in range(max_sweeps):
            improved = False
            deltas = all_deltas(cols_np)
            neg_mask = deltas < 0
            if not np.any(neg_mask):
                break
            for i in np.where(neg_mask)[0]:
                d = deltas[i]
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
            if not improved:
                break
        return cols, current_cost
    
    def thue_morse_coloring():
        colors = [0] * N
        for i in range(N):
            n = i + 1
            colors[i] = bin(n).count('1') % 2
        return colors
    
    def generate_structured_patterns():
        patterns = []
        patterns.append(thue_morse_coloring())
        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])
        for block in [2, 3, 4, 5, 7, 8, 13, 16, 32, 64]:
            patterns.append([(i // block) % 2 for i in range(N)])
        patterns.append([i % 2 for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 2) & 1) for i in range(N)])
        patterns.append([(i & 3) ^ ((i >> 2) & 3) for i in range(N)])
        for seed in range(5):
            rng = random.Random(seed)
            patterns.append([rng.randint(0, 1) for _ in range(N)])
        
        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:10]]
    
    # Get initial patterns
    initial_patterns = generate_structured_patterns()
    
    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))
    
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit, seed):
        rng = random.Random(seed)
        colors = initial_colors[:]
        cols_np = np.array(colors, dtype=np.int8)
        current_cost = count_mono(cols_np)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = rng.randint(0, N - 1)
            delta = all_deltas(cols_np)[i]
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                if rng.random() < 0.5:
                    colors[i] = 1 - colors[i]
                    cols_np[i] = 1 - cols_np[i]
            else:
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        colors[i] = 1 - colors[i]
                        cols_np[i] = 1 - cols_np[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    import math
    
    # Multiple restarts with simulated annealing + polish
    num_restarts = 15
    time_per_restart = 135.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose initial pattern
        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 10:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 4, N // 3)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 8, N // 4)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        
        T_start = 2.0 + 0.5 * (restart % 4)
        cooling = 0.99998
        min_T = 0.01
        seed = 42 + restart * 131
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.7, seed
        )
        
        # Polish
        batch_colors, batch_cost = batch_polish(sa_colors, sa_cost, 3.0)
        polished_colors, polished_cost = greedy_polish(batch_colors, batch_cost, max_sweeps=30)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish
    if time.time() < deadline - 5:
        final_colors, final_cost = batch_polish(global_best_colors[:], global_best_cost, 5.0)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
        final_colors2, final_cost2 = greedy_polish(global_best_colors[:], global_best_cost, max_sweeps=50)
        if final_cost2 < global_best_cost:
            global_best_cost = final_cost2
            global_best_colors = final_colors2
    
    return global_best_colors
