import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    def count_mono_np(x):
        """x is np.array of 0s and 1s (int32)."""
        vals = x[ap_indices]
        # monochromatic if all 0 or all 1
        # all 0: prod(1-vals) ; all 1: prod(vals)
        p1 = np.prod(vals, axis=1)
        p0 = np.prod(1 - vals, axis=1)
        return int(np.sum(p1 + p0))

    # For fast delta of single flip at position k:
    # We need, for each AP containing k, the colors of the other 3 positions.
    # Precompute for each k the list of (other_idx1, other_idx2, other_idx3)
    link_info = [[] for _ in range(N)]
    for ap in aps:
        for pos in range(4):
            idx = ap[pos]
            others = tuple(ap[k] for k in range(4) if k != pos)
            link_info[idx].append(others)

    # For pair flip delta, precompute common APs
    common_aps = {}
    for ap_idx, ap in enumerate(aps):
        for k in range(4):
            for l in range(k+1, 4):
                i, j = ap[k], ap[l]
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key not in common_aps:
                    common_aps[key] = []
                common_aps[key].append(ap_idx)

    bit_to_ap_indices = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            bit_to_ap_indices[idx].append(ap_idx)

    def delta_flip(colors, k):
        """colors is a Python list or numpy array of 0/1."""
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            if c1 == c2 == c3:
                if c1 == old_c:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def eval_pair_flip(colors, i, j):
        if i > j:
            i, j = j, i
        old_i = colors[i]
        old_j = colors[j]
        new_i = 1 - old_i
        new_j = 1 - old_j
        delta = 0
        
        aps_i_set = set(bit_to_ap_indices[i])
        aps_j_set = set(bit_to_ap_indices[j])
        common_set = set(common_aps.get((i, j), []))
        
        # APs with only i
        for ap_idx in aps_i_set:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            others = [idx for idx in ap if idx != i]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3:
                if c1 == old_i:
                    delta -= 1
                else:
                    delta += 1
        
        # APs with only j
        for ap_idx in aps_j_set:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            others = [idx for idx in ap if idx != j]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3:
                if c1 == old_j:
                    delta -= 1
                else:
                    delta += 1
        
        # APs with both i and j
        for ap_idx in common_set:
            ap = aps[ap_idx]
            others = [idx for idx in ap if idx != i and idx != j]
            c_k = colors[others[0]]
            c_l = colors[others[1]]
            if c_k == c_l:
                if old_i == old_j == c_k:
                    delta -= 1
                elif new_i == new_j == c_k:
                    delta += 1
        return delta

    def local_search(colors, cost, max_iters=20000):
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            if time.time() > deadline - 5.0:
                break
            improved = False
            iterations += 1
            
            # Try all single flips
            for i in range(N):
                d = delta_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    cost += d
                    improved = True
            if improved:
                continue
            
            # Try random pair flips
            for _ in range(2000):
                if time.time() > deadline - 5.0:
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
        t0 = time.time()
        best_colors = init_colors[:]
        best_cost = init_cost
        current_colors = init_colors[:]
        current_cost = init_cost
        
        while time.time() < t0 + time_limit:
            current_colors, current_cost = local_search(current_colors, current_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors[:]
            
            if time.time() > t0 + time_limit - 3.0:
                break
            
            # Perturb: flip random number of bits
            num_flips = random.randint(3, 12)
            indices = random.sample(range(N), num_flips)
            for idx in indices:
                current_colors[idx] = 1 - current_colors[idx]
            current_cost = count_mono_np(np.array(current_colors, dtype=np.int32))
        
        return best_colors, best_cost

    # Generate initializations
    inits = []
    
    # Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    inits.append([thue_morse(i) for i in range(N)])
    
    # All zeros / all ones (bad but include for completeness)
    inits.append([0] * N)
    inits.append([1] * N)
    
    # Random with fixed seeds
    random.seed(42)
    for _ in range(15):
        inits.append([random.randint(0, 1) for _ in range(N)])
    
    # Block patterns
    for bs in [2, 4, 8, 16, 32, 64, 128]:
        inits.append([(i // bs) % 2 for i in range(N)])
    
    # Alternating
    inits.append([i % 2 for i in range(N)])
    
    # Evaluate all inits
    best_init = None
    best_init_cost = float('inf')
    for init in inits:
        c = count_mono_np(np.array(init, dtype=np.int32))
        if c < best_init_cost:
            best_init_cost = c
            best_init = init[:]
    
    global_best_colors = best_init[:]
    global_best_cost = best_init_cost
    
    # Run ILS with remaining time
    remaining = deadline - time.time() - 3.0
    if remaining > 10.0:
        global_best_colors, global_best_cost = iterated_local_search(
            best_init, best_init_cost, remaining
        )
    
    # Final polish: greedy single flips
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for pos in range(N):
                d = delta_flip(global_best_colors, pos)
                if d < 0:
                    global_best_colors[pos] = 1 - global_best_colors[pos]
                    global_best_cost += d
                    improved = True
    
    return global_best_colors
