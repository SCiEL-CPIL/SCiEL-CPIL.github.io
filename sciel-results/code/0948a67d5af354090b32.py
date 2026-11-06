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
    
    # For each position i, precompute:
    # - ap_indices[i]: indices of APs containing position i
    # - For each such AP, the other 3 positions
    # Store as arrays for vectorized delta computation
    ap_indices_per_pos = [[] for _ in range(N)]
    other_positions_per_pos = [[] for _ in range(N)]
    
    for idx, ap in enumerate(aps):
        for j in range(4):
            pos = ap[j]
            others = [ap[k] for k in range(4) if k != j]
            ap_indices_per_pos[pos].append(idx)
            other_positions_per_pos[pos].append(others)
    
    # Convert to numpy arrays for fast access
    ap_idx_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_indices_per_pos]
    other_pos_arrays = [np.array(lst, dtype=np.int32).reshape(-1, 3) if lst else np.array([[]]).reshape(0, 3) for lst in other_positions_per_pos]
    
    # Also store APs as lists for delta_flip_2
    ap_list = [list(ap) for ap in aps]
    
    def count_mono(x):
        x = np.asarray(x, dtype=np.int8)
        vals = x[ap_array]
        return int(np.sum(np.all(vals == vals[:, 0:1], axis=1)))
    
    def delta_flip_1_vec(x, i):
        """Vectorized delta for flipping position i."""
        if len(ap_idx_arrays[i]) == 0:
            return 0
        old_val = x[i]
        new_val = 1 - old_val
        
        # Get all APs containing position i
        ap_idxs = ap_idx_arrays[i]
        others = other_pos_arrays[i]  # shape (num_aps_containing_i, 3)
        
        # Get values of the other 3 positions for each AP
        # others has shape (k, 3), we need x[others] which gives (k, 3)
        other_vals = x[others]  # shape (k, 3)
        
        # was_mono: all 3 others equal old_val
        was_mono = np.all(other_vals == old_val, axis=1)  # shape (k,)
        # is_mono: all 3 others equal new_val  
        is_mono = np.all(other_vals == new_val, axis=1)  # shape (k,)
        
        delta = int(np.sum(~was_mono & is_mono) - np.sum(was_mono & ~is_mono))
        return delta
    
    def delta_flip_2(x, i, j):
        if i == j:
            return 0
        old_i = x[i]
        old_j = x[j]
        new_i = 1 - old_i
        new_j = 1 - old_j
        
        # Get affected APs (union of APs containing i or j)
        affected_aps = set()
        for ap_idx in ap_idx_arrays[i]:
            affected_aps.add(ap_idx)
        for ap_idx in ap_idx_arrays[j]:
            affected_aps.add(ap_idx)
            
        delta = 0
        for ap_idx in affected_aps:
            ap = ap_list[ap_idx]
            v0 = x[ap[0]]
            v1 = x[ap[1]]
            v2 = x[ap[2]]
            v3 = x[ap[3]]
            
            was_mono = (v0 == v1 == v2 == v3)
            
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
    
    def simulated_annealing(initial_x, time_limit, T_start=5.0, T_end=0.005):
        x = np.array(initial_x, dtype=np.int8)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        # Use a faster cooling rate to actually anneal within time budget
        # With ~1M steps in 100s, we want T to go from 5 to 0.005
        # That's a factor of 1000, so per step: 1000^(-1/1M) ≈ 0.999993
        # But let's be more aggressive: 0.9999 per step
        cool_rate = 0.9999
        
        while time.time() < time_limit:
            # Choose move: mostly 1-flips (faster), some 2-flips
            if random.random() < 0.2:
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
                # 1-flip using vectorized delta
                i = random.randint(0, N - 1)
                d = delta_flip_1_vec(x, i)
                if d < 0 or random.random() < np.exp(-d / T):
                    x[i] = 1 - x[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
            
            # Cooling
            step += 1
            T *= cool_rate
            if T < T_end:
                T = T_end
                
        return best_x, best_cost
    
    def greedy_polish(x, time_limit):
        x = np.array(x, dtype=np.int8)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                d = delta_flip_1_vec(x, i)
                if d < 0:
                    x[i] = 1 - x[i]
                    improved = True
        return x
    
    def perturb(x, num_flips=4):
        """Perturb the solution by flipping a few random positions."""
        x_pert = x.copy()
        for _ in range(num_flips):
            i = random.randint(0, N - 1)
            x_pert[i] = 1 - x_pert[i]
        return x_pert
    
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
        np.array(periodic(N, 3), dtype=np.int8),
    ]
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run multiple SA rounds with perturbation-restart strategy
    # Phase 1: Try different initializations
    num_inits = len(inits)
    phase1_time = (deadline - start_time) * 0.4
    chunk_time = phase1_time / num_inits
    
    for idx, init in enumerate(inits):
        t_limit = start_time + (idx + 1) * chunk_time
        if time.time() >= t_limit:
            continue
        x_best, c_best = simulated_annealing(init, t_limit)
        x_polished = greedy_polish(x_best, t_limit + 2.0)
        c_polished = count_mono(x_polished)
        if c_polished < best_overall_cost:
            best_overall_cost = c_polished
            best_overall_x = x_polished.copy()
            
    # Phase 2: Perturbation-restart on best solution
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10.0:
        num_restarts = 6
        restart_time = remaining_time / num_restarts
        
        for r in range(num_restarts):
            t_limit = time.time() + restart_time
            if time.time() >= t_limit - 1.0:
                break
            
            # Perturb the best solution
            x_pert = perturb(best_overall_x, num_flips=3 + r)
            x_best, c_best = simulated_annealing(x_pert, t_limit, T_start=2.0, T_end=0.002)
            x_polished = greedy_polish(x_best, t_limit + 1.0)
            c_polished = count_mono(x_polished)
            if c_polished < best_overall_cost:
                best_overall_cost = c_polished
                best_overall_x = x_polished.copy()
    
    # Final polish if time remains
    if time.time() < deadline - 2.0 and best_overall_x is not None:
        x_final = greedy_polish(best_overall_x, deadline - 1.0)
        c_final = count_mono(x_final)
        if c_final < best_overall_cost:
            best_overall_cost = c_final
            best_overall_x = x_final
    
    return [int(v) for v in best_overall_x]
