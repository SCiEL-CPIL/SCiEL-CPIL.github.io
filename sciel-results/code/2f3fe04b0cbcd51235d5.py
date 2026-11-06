import time
import random
import math
import numpy as np

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # NumPy array for vectorized full evaluation
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_mono_np(x):
        """Vectorized count using NumPy. x is np.array of 0s and 1s."""
        vals = x[ap_indices]  # shape (num_aps, 4)
        prod1 = np.prod(vals, axis=1)
        prod0 = np.prod(1 - vals, axis=1)
        return int(np.sum(prod1 + prod0))
    
    # Precompute link_info for fast delta_flip (single bit)
    link_info = [[] for _ in range(N)]
    for ap in aps:
        for pos in range(4):
            idx = ap[pos]
            others = tuple(ap[k] for k in range(4) if k != pos)
            link_info[idx].append(others)
    
    # Precompute bit_to_ap_indices
    bit_to_ap_indices = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            bit_to_ap_indices[idx].append(ap_idx)
    
    # Precompute common_aps for pair flips (APs containing both i and j)
    common_aps = {}
    for ap_idx, ap in enumerate(aps):
        for k in range(4):
            for l in range(k+1, 4):
                i = ap[k]
                j = ap[l]
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key not in common_aps:
                    common_aps[key] = []
                common_aps[key].append(ap_idx)

    def delta_flip(colors, k):
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == new_c:
                delta += 1
            if c1 == c2 == c3 == old_c:
                delta -= 1
        return delta
        
    def eval_pair_flip(colors, i, j):
        if i > j:
            i, j = j, i
            
        old_i = colors[i]
        old_j = colors[j]
        new_i = 1 - old_i
        new_j = 1 - old_j
        
        delta = 0
        
        # Get sets of AP indices for i and j
        aps_i = set(bit_to_ap_indices[i])
        aps_j = set(bit_to_ap_indices[j])
        common_set = set(common_aps.get((i, j), []))
        
        # APs containing only i
        for ap_idx in aps_i:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            # Find the other 3 indices
            others = [idx for idx in ap if idx != i]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == old_i:
                delta -= 1
            elif c1 == c2 == c3 == new_i:
                delta += 1
                
        # APs containing only j
        for ap_idx in aps_j:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            others = [idx for idx in ap if idx != j]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == old_j:
                delta -= 1
            elif c1 == c2 == c3 == new_j:
                delta += 1
                
        # APs containing both i and j
        for ap_idx in common_set:
            ap = aps[ap_idx]
            others = [idx for idx in ap if idx != i and idx != j]
            c_k, c_l = colors[others[0]], colors[others[1]]
            if old_i == old_j == c_k == c_l:
                delta -= 1
            elif new_i == new_j == c_k == c_l:
                delta += 1
                
        return delta

    def local_search(colors, cost, max_iters=10000, time_limit=None, start_time=None):
        if start_time is None:
            start_time = time.time()
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            if time_limit and time.time() > start_time + time_limit:
                break
            improved = False
            iterations += 1
            
            # Try single flips
            for i in range(N):
                d = delta_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    cost += d
                    improved = True
            if improved:
                continue
                
            # Try pair flips
            # Sample random pairs to speed up
            num_pairs_to_try = 1000
            for _ in range(num_pairs_to_try):
                if time_limit and time.time() > start_time + time_limit:
                    break
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                d = eval_pair_flip(colors, i, j)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    cost += d
                    improved = True
                    break
                    
        return colors, cost

    def iterated_local_search(init_colors, init_cost, time_limit):
        start_time = time.time()
        best_colors = init_colors[:]
        best_cost = init_cost
        
        current_colors = init_colors[:]
        current_cost = init_cost
        
        while time.time() < start_time + time_limit:
            # Local Search
            current_colors, current_cost = local_search(current_colors, current_cost, time_limit=time_limit, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors[:]
                
            # Perturb
            # Flip a random number of bits
            num_flips = random.randint(5, 15)
            indices = random.sample(range(N), num_flips)
            for idx in indices:
                current_colors[idx] = 1 - current_colors[idx]
            current_cost = count_mono_np(np.array(current_colors, dtype=np.int32))
            
        return best_colors, best_cost

    # Main
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate diverse initializations
    inits = []
    
    # Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    tm = [thue_morse(i) for i in range(N)]
    inits.append(tm)
    
    # Random
    random.seed(42)
    for _ in range(10):
        inits.append([random.randint(0, 1) for _ in range(N)])
        
    # Block patterns
    for block_size in [4, 8, 16, 32, 64]:
        inits.append([(i // block_size) % 2 for i in range(N)])
        
    # Evaluate and pick best init
    best_init = None
    best_init_cost = float('inf')
    for init in inits:
        cost = count_mono_np(np.array(init, dtype=np.int32))
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = init[:]
            
    global_best_colors = best_init[:]
    global_best_cost = best_init_cost
    
    # Phase 1: Run ILS for the first portion of time
    time_for_ils = (deadline - start_time) * 0.4
    global_best_colors, global_best_cost = iterated_local_search(best_init, best_init_cost, time_for_ils)
    
    # Phase 2: Simulated Annealing with multi-flip moves (vectorized subgradient-style updates)
    # This allows larger moves that can escape local minima more effectively
    sa_start = time.time()
    sa_deadline = deadline - 5.0  # Leave 5s for final polish
    
    # Convert to numpy for potential vectorized operations
    current_colors = np.array(global_best_colors, dtype=np.int32)
    current_cost = global_best_cost
    
    # SA parameters
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    steps_since_improve = 0
    max_steps_no_improve = 50000
    
    while time.time() < sa_deadline:
        # Choose move size: 1, 2, or 3 bits
        move_size = random.choice([1, 1, 1, 2, 2, 3])
        
        if move_size == 1:
            k = random.randint(0, N - 1)
            d_cost = delta_flip(current_colors.tolist(), k)
            if d_cost < 0 or (T > T_min and random.random() < math.exp(-d_cost / T)):
                current_colors[k] = 1 - current_colors[k]
                current_cost += d_cost
        elif move_size == 2:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            colors_list = current_colors.tolist()
            d_cost = eval_pair_flip(colors_list, i, j)
            if d_cost < 0 or (T > T_min and random.random() < math.exp(-d_cost / T)):
                current_colors[i] = 1 - current_colors[i]
                current_colors[j] = 1 - current_colors[j]
                current_cost += d_cost
        else:  # move_size == 3
            # Triple flip: evaluate by brute force delta
            indices = random.sample(range(N), 3)
            colors_list = current_colors.tolist()
            # Compute cost before
            old_cost = current_cost
            # Flip all three
            for idx in indices:
                colors_list[idx] = 1 - colors_list[idx]
            new_cost = count_mono_np(np.array(colors_list, dtype=np.int32))
            d_cost = new_cost - old_cost
            if d_cost < 0 or (T > T_min and random.random() < math.exp(-d_cost / T)):
                current_colors = np.array(colors_list, dtype=np.int32)
                current_cost = new_cost
            else:
                # Revert
                for idx in indices:
                    colors_list[idx] = 1 - colors_list[idx]
                current_colors = np.array(colors_list, dtype=np.int32)
        
        # Track best
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_colors = current_colors.tolist()
            steps_since_improve = 0
        else:
            steps_since_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # If stuck at low temperature, restart with perturbation
        if T <= T_min and steps_since_improve > max_steps_no_improve:
            # Perturb current solution
            num_flips = random.randint(10, 20)
            flip_indices = random.sample(range(N), num_flips)
            for idx in flip_indices:
                current_colors[idx] = 1 - current_colors[idx]
            current_cost = count_mono_np(current_colors)
            # Reset temperature
            T = 1.0
            steps_since_improve = 0
    
    # Phase 3: Final greedy polish with single and pair flips
    if time.time() < deadline - 2.0:
        improved = True
        polish_deadline = deadline - 2.0
        while improved and time.time() < polish_deadline:
            improved = False
            # Single flips
            for pos in range(N):
                if time.time() > polish_deadline:
                    break
                d = delta_flip(global_best_colors, pos)
                if d < 0:
                    global_best_colors[pos] = 1 - global_best_colors[pos]
                    global_best_cost += d
                    improved = True
            # Pair flips
            if not improved:
                for _ in range(2000):
                    if time.time() > polish_deadline:
                        break
                    i = random.randint(0, N - 1)
                    j = random.randint(0, N - 1)
                    if i == j:
                        continue
                    d = eval_pair_flip(global_best_colors, i, j)
                    if d < 0:
                        global_best_colors[i] = 1 - global_best_colors[i]
                        global_best_colors[j] = 1 - global_best_colors[j]
                        global_best_cost += d
                        improved = True
                        break
    
    return global_best_colors
