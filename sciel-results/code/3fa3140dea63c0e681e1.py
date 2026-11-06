import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Map bit index to list of (ap_idx, other_pos1, other_pos2, other_pos3)
    bit_ap_data = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j in range(4):
            pos = ap[j]
            others = [ap[k] for k in range(4) if k != j]
            bit_ap_data[pos].append((idx, others[0], others[1], others[2]))
            
    # Also store APs as lists for faster access in delta calc if needed
    ap_list = [list(ap) for ap in aps]

    def count_mono(x):
        x = np.asarray(x, dtype=np.int8)
        vals = x[ap_array]
        return int(np.sum(np.all(vals == vals[:, 0:1], axis=1)))

    def delta_flip_1(x, i):
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for (ap_idx, o1, o2, o3) in bit_ap_data[i]:
            # Check if other 3 are old_val (was mono)
            was_mono = (x[o1] == old_val) and (x[o2] == old_val) and (x[o3] == old_val)
            # Check if other 3 are new_val (becomes mono)
            is_mono = (x[o1] == new_val) and (x[o2] == new_val) and (x[o3] == new_val)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    def delta_flip_2(x, i, j):
        if i == j:
            return 0
        old_i = x[i]
        old_j = x[j]
        new_i = 1 - old_i
        new_j = 1 - old_j
        
        # We need to check all APs containing i or j
        # To avoid double counting, we can iterate through the union
        # But simple approach: iterate through bit_ap_data[i] and bit_ap_data[j]
        # and use a set of ap indices to avoid duplicates.
        # However, sets are slow. 
        # Alternative: just iterate through both lists, and if an AP is in both, 
        # it will be processed twice? No, we need to process it once.
        # Let's use a set of ap indices.
        affected_aps = set()
        for (ap_idx, _, _, _) in bit_ap_data[i]:
            affected_aps.add(ap_idx)
        for (ap_idx, _, _, _) in bit_ap_data[j]:
            affected_aps.add(ap_idx)
            
        delta = 0
        for ap_idx in affected_aps:
            ap = ap_list[ap_idx]
            # Get values of the 4 positions in current state x
            v0 = x[ap[0]]
            v1 = x[ap[1]]
            v2 = x[ap[2]]
            v3 = x[ap[3]]
            
            # Check if mono before
            was_mono = (v0 == v1 == v2 == v3)
            
            # Calculate new values
            # If ap[0] is i, new value is new_i, else v0
            # This is O(1) per AP
            nv0 = new_i if ap[0] == i else (new_j if ap[0] == j else v0)
            nv1 = new_i if ap[1] == i else (new_j if ap[1] == j else v1)
            nv2 = new_i if ap[2] == i else (new_j if ap[2] == j else v2)
            nv3 = new_i if ap[3] == i else (new_j if ap[3] == j else v3)
            
            is_mono = (nv0 == nv1 == nv2 == nv3)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    def simulated_annealing(initial_x, time_limit, T_start=10.0, T_end=0.01):
        x = np.array(initial_x, dtype=np.int8)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        # Estimate steps: delta_flip_1 is ~50 ops. delta_flip_2 is ~100 ops.
        # Let's assume 100k steps in 10s. 
        # We don't know exact time limit per run yet, so we just loop until time_limit.
        
        while time.time() < time_limit:
            # Choose move
            if random.random() < 0.4:
                # 2-flip
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                d = delta_flip_2(x, i, j)
                if d < 0 or random.random() < np.exp(-d / T):
                    x[i] = 1 - x[i]
                    x[j] = 1 - x[j]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
            else:
                # 1-flip
                i = random.randint(0, N - 1)
                d = delta_flip_1(x, i)
                if d < 0 or random.random() < np.exp(-d / T):
                    x[i] = 1 - x[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
            
            # Cooling
            step += 1
            # Adaptive cooling: reduce T slightly each step
            T *= 0.99995
            if T < T_end:
                T = T_end
                
        return best_x, best_cost

    def greedy_polish(x, time_limit):
        x = np.array(x, dtype=np.int8)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Random order to avoid bias? Or fixed order. Fixed is faster.
            for i in range(N):
                d = delta_flip_1(x, i)
                if d < 0:
                    x[i] = 1 - x[i]
                    improved = True
        return x

    # Initializations
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def random_pattern(n):
        return [random.randint(0, 1) for _ in range(n)]
        
    def periodic(n, p):
        return [i % p % 2 for i in range(n)]

    inits = [
        np.array(random_pattern(N), dtype=np.int8),
        np.array(thue_morse(N), dtype=np.int8),
        np.array(periodic(N, 4), dtype=np.int8),
        np.array(periodic(N, 5), dtype=np.int8),
    ]
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run SA for a few iterations
    # Split time into 4 chunks
    chunk_time = (deadline - start_time) / 4.0
    
    for idx, init in enumerate(inits):
        t_limit = start_time + (idx + 1) * chunk_time
        if time.time() >= t_limit:
            continue
        x_best, c_best = simulated_annealing(init, t_limit)
        # Polish
        x_polished = greedy_polish(x_best, t_limit + 5.0)
        c_polished = count_mono(x_polished)
        if c_polished < best_overall_cost:
            best_overall_cost = c_polished
            best_overall_x = x_polished
            
    # If time remains, do more SA on the best found
    if time.time() < deadline - 10.0:
        x_final, c_final = simulated_annealing(best_overall_x, deadline - 10.0, T_start=1.0, T_end=0.001)
        x_polished = greedy_polish(x_final, deadline)
        c_polished = count_mono(x_polished)
        if c_polished < best_overall_cost:
            best_overall_cost = c_polished
            best_overall_x = x_polished

    return [int(v) for v in best_overall_x]
