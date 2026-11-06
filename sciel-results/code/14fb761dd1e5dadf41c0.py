import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)

    # Convert to numpy array for vectorized operations
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # Precompute for each bit, the AP indices it participates in (as numpy arrays for vectorized ops)
    # bit_ap_indices[i] = array of AP indices containing bit i
    # bit_others[i] = array of shape (k_i, 3) with the other 3 positions for each AP containing bit i
    bit_ap_indices = [None] * N
    bit_others = [None] * N
    for i in range(N):
        idxs = []
        oth = []
        for ap_idx, ap in enumerate(aps):
            if i in ap:
                pos_in_ap = ap.index(i)
                others = [ap[k] for k in range(4) if k != pos_in_ap]
                idxs.append(ap_idx)
                oth.append(others)
        bit_ap_indices[i] = np.array(idxs, dtype=np.int32)
        bit_others[i] = np.array(oth, dtype=np.int32) if oth else np.empty((0, 3), dtype=np.int32)

    # Function to count monochromatic APs (vectorized)
    def count_mono(x):
        x = np.asarray(x, dtype=np.int8)
        vals = x[ap_array]  # (num_aps, 4)
        return int(np.sum(np.all(vals == vals[:, 0:1], axis=1)))

    # Vectorized delta for flipping bit i
    def delta_flip(x, i):
        old_val = x[i]
        new_val = 1 - old_val
        others = bit_others[i]
        if len(others) == 0:
            return 0
        # x[others] has shape (k, 3)
        vals = x[others]  # (k, 3)
        # was_mono: all 3 others == old_val
        was_mono = np.all(vals == old_val, axis=1)  # (k,)
        # is_mono: all 3 others == new_val
        is_mono = np.all(vals == new_val, axis=1)  # (k,)
        # delta = count(was_mono & ~is_mono) - count(~was_mono & is_mono)
        # = sum(was_mono) - sum(is_mono)
        delta = int(np.sum(was_mono)) - int(np.sum(is_mono))
        return delta

    # Simulated Annealing with vectorized delta
    def simulated_annealing(initial_x, time_limit, T_start=5.0, T_end=0.05):
        x = np.array(initial_x, dtype=np.int8)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        total_steps_estimate = 300000
        cool_rate = (T_end / T_start) ** (1.0 / total_steps_estimate)
        
        while time.time() < time_limit:
            # 1-flip move
            i = random.randint(0, N - 1)
            d = delta_flip(x, i)
            
            # Accept if improvement or with probability exp(-d/T)
            if d < 0 or random.random() < np.exp(-d / T):
                x[i] = 1 - x[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            
            # Cool down
            step += 1
            T *= cool_rate
            if T < T_end:
                T = T_end
        
        return best_x, best_cost

    # Initialize with diverse starting points
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def periodic_pattern(n, period):
        return [i % period % 2 for i in range(n)]
    
    tm_x = np.array(thue_morse(N), dtype=np.int8)
    rand_x = np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)
    per4_x = np.array(periodic_pattern(N, 4), dtype=np.int8)
    per5_x = np.array([i % 5 < 2 for i in range(N)], dtype=np.int8)
    # Additional diverse starts
    per3_x = np.array([i % 3 < 2 for i in range(N)], dtype=np.int8)
    per6_x = np.array([i % 6 < 3 for i in range(N)], dtype=np.int8)
    # Block pattern: 000111000111...
    block6_x = np.array([((i // 3) % 2) for i in range(N)], dtype=np.int8)
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run SA on multiple starts, splitting time
    starts = [rand_x, tm_x, per4_x, per5_x, per3_x, per6_x, block6_x]
    n_starts = len(starts)
    time_per_start = (deadline - start_time) / n_starts
    
    for s_idx, init_x in enumerate(starts):
        t_limit = start_time + (s_idx + 1) * time_per_start
        if time.time() < t_limit:
            x_best, c_best = simulated_annealing(init_x, t_limit)
            if c_best < best_overall_cost:
                best_overall_cost = c_best
                best_overall_x = x_best
    
    # Fallback
    if best_overall_x is None:
        best_overall_x = rand_x
    
    # Final polish: greedy local search on best solution (vectorized)
    x = np.array(best_overall_x, dtype=np.int8)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(N):
            d = delta_flip(x, i)
            if d < 0:
                x[i] = 1 - x[i]
                improved = True
    
    # Check if polished solution is better
    final_cost = count_mono(x)
    if final_cost < best_overall_cost:
        best_overall_x = x
        best_overall_cost = final_cost
    
    # Return as list of ints
    return [int(v) for v in best_overall_x]
