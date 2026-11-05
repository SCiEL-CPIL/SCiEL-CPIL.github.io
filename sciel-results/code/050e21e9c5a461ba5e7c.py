import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 140
    
    # Precompute all 4-APs as numpy arrays
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))
    
    aps_np = np.array(aps_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps_np)
    
    # For each position, precompute which APs it belongs to and the other 3 positions
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))
    
    # Convert to numpy arrays for fast lookup
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))
    
    # Precompute the count of APs per position for weighted random selection
    ap_counts = np.array([len(pos_aps[i]) for i in range(N)], dtype=np.float64)
    ap_probs = ap_counts / ap_counts.sum()
    
    def count_mono(cols_np):
        """Vectorized count of monochromatic 4-APs."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def delta_for_flip(cols_np, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols_np[i]
        arr = pos_aps_arr[i]
        if len(arr) == 0:
            return 0
        # arr shape: (k, 4) = [ap_idx, other1, other2, other3]
        others = arr[:, 1:]  # shape (k, 3)
        c_others = cols_np[others]  # shape (k, 3)
        # Check if all 3 others are the same color
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        n_same = int(np.sum(all_same))
        if n_same == 0:
            return 0
        # For those where all 3 are same, check if they match c_i
        matching = int(np.sum(c_others[all_same, 0] == c_i))
        # If we flip: APs where others all match c_i become non-mono (lose matching)
        # APs where others all differ from c_i become mono (gain n_same - matching)
        return -(matching) + (n_same - matching)
    
    def greedy_polish(cols, current_cost):
        """Try all single flips and apply any that reduce cost. Iterate until no improvement."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(cols_np, i)
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
        return cols, current_cost
    
    def thue_morse_coloring():
        """Generate Thue-Morse coloring (1-indexed)."""
        colors = [0] * N
        for i in range(N):
            n = i + 1  # 1-indexed
            # Count bits in binary representation
            colors[i] = bin(n).count('1') % 2
        return colors
    
    def thue_morse_shifted(shift):
        """Thue-Morse with a shift."""
        colors = [0] * N
        for i in range(N):
            n = (i + 1 + shift) % (N + 1)
            if n == 0:
                n = N + 1
            colors[i] = bin(n).count('1') % 2
        return colors
    
    def generate_structured_patterns():
        """Generate diverse structured initial colorings."""
        patterns = []
        
        # Thue-Morse
        patterns.append(thue_morse_coloring())
        
        # Thue-Morse shifted
        for shift in range(1, 5):
            patterns.append(thue_morse_shifted(shift))
        
        # Inverted Thue-Morse
        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])
        
        # Block patterns
        for block in [2, 4, 8]:
            p = [(i // block) % 2 for i in range(N)]
            patterns.append(p)
        
        # Alternating
        patterns.append([i % 2 for i in range(N)])
        
        # XOR pattern
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        
        # Random
        rng_init = random.Random(12345)
        for _ in range(2):
            patterns.append([rng_init.randint(0, 1) for _ in range(N)])
        
        # Evaluate all and return top 4
        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:4]]
    
    # Get initial patterns
    initial_patterns = generate_structured_patterns()
    
    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))
    
    # Simulated annealing
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        colors = initial_colors[:]
        cols_np = np.array(colors, dtype=np.int8)
        current_cost = count_mono(cols_np)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        # Use weighted random selection: positions in more APs are more impactful
        while time.time() < end_time:
            i = rng.choices(range(N), weights=ap_probs, k=1)[0]
            delta = delta_for_flip(cols_np, i)
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                if rng.random() < 0.3:
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
    
    # Multiple restarts
    num_restarts = 12
    time_per_restart = 120.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Choose initial
        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 6:
            # Perturb best
            initial = global_best_colors[:]
            n_flips = random.randint(N // 4, N // 2)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            # Moderate perturbation of best
            initial = global_best_colors[:]
            n_flips = random.randint(N // 8, N // 4)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        
        # SA with varying parameters
        T_start = 1.5 + 0.5 * (restart % 4)
        cooling = 0.99995
        min_T = 0.003
        seed = 42 + restart * 137
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.8, seed=seed
        )
        
        # Greedy polish
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish
    if time.time() < deadline - 5:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
    
    return global_best_colors
