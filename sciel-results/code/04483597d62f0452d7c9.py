import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0

    # Precompute all 4-term APs
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # For each element, store which APs it appears in and at what position
    x_ap_pos = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for pos, x in enumerate(ap):
            x_ap_pos[x].append((idx, pos))
    
    # Precompute for each AP and position, the other 3 elements
    ap_others = []
    for ap in all_aps:
        others = []
        for i in range(4):
            o = [ap[j] for j in range(4) if j != i]
            others.append(tuple(o))
        ap_others.append(others)
    
    # Bit-parallel representation: each element x has a bitmask of all 4-APs it participates in
    # For conflict detection: x can be added iff for every AP containing x, not all other 3 are present
    # We precompute for each x: a list of (ap_idx, pos) as before, but also build a "conflict mask"
    # approach: for each x, precompute the set of "triple masks" - for each AP containing x at position pos,
    # create a 3-element mask of the other elements. Then x conflicts iff any such triple is fully present.
    
    # More efficient: use numpy boolean array and vectorized checks
    # For each x, precompute arrays of the other elements for fast vectorized checking
    
    # x_conflict_data[x] = (ap_indices, positions, other1_arr, other2_arr, other3_arr)
    x_conflict_data = [None] * (N + 1)
    for x in range(1, N + 1):
        if not x_ap_pos[x]:
            x_conflict_data[x] = (None, None, None, None)
            continue
        ap_indices = np.array([ap_idx for ap_idx, pos in x_ap_pos[x]], dtype=np.int32)
        positions = np.array([pos for ap_idx, pos in x_ap_pos[x]], dtype=np.int32)
        
        # For each (ap_idx, pos), get the other 3 elements
        o1 = np.array([ap_others[ap_idx][pos][0] for ap_idx, pos in x_ap_pos[x]], dtype=np.int32)
        o2 = np.array([ap_others[ap_idx][pos][1] for ap_idx, pos in x_ap_pos[x]], dtype=np.int32)
        o3 = np.array([ap_others[ap_idx][pos][2] for ap_idx, pos in x_ap_pos[x]], dtype=np.int32)
        
        x_conflict_data[x] = (o1, o2, o3)
    
    def can_add_fast(x, in_set_np):
        """Check if x can be added using vectorized numpy operations"""
        o1, o2, o3 = x_conflict_data[x]
        if o1 is None:
            return True
        # Check if any AP has all 3 other elements present
        conflict = (in_set_np[o1] & in_set_np[o2] & in_set_np[o3])
        return not conflict.any()
    
    def set_to_np(S):
        in_set = np.zeros(N + 1, dtype=bool)
        for x in S:
            in_set[x] = True
        return in_set
    
    def np_to_set(in_set_np):
        return {i for i in range(1, N + 1) if in_set_np[i]}
    
    # Initial Heuristics
    best_set = set()
    best_size = 0
    
    # 1. Base-4 construction
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
            
    if len(base4_set) > best_size:
        best_set = base4_set
        best_size = len(base4_set)
        
    # 2. Multiple randomized greedy initializations with vectorized checks
    for seed in range(60):
        if time.time() > deadline - 40:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = set()
        in_set = np.zeros(N + 1, dtype=bool)
        for x in order:
            if can_add_fast(x, in_set):
                S.add(x)
                in_set[x] = True
        if len(S) > best_size:
            best_set = S
            best_size = len(S)

    # 3. Simulated Annealing / Iterated Local Search with vectorized checks
    rng = random.Random(12345)
    current_set = set(best_set)
    current_in_set = set_to_np(current_set)
    current_size = best_size
    
    best_set = set(current_set)
    best_size = current_size
    
    # Temperature parameters
    T_initial = 2.0
    T_min = 0.05
    cooling_rate = 0.997
    
    stuck_count = 0
    T = T_initial
    
    while time.time() < deadline - 2:
        # Choose neighborhood size
        if stuck_count > 80:
            k = rng.randint(3, 6)
            stuck_count = 0
        else:
            k = rng.randint(1, 3)
        
        # Create perturbation
        S_curr = set(current_set)
        in_set_curr = set_to_np(S_curr)
        
        # Remove k random elements
        elements = list(S_curr)
        rng.shuffle(elements)
        to_remove = elements[:k]
        
        for x in to_remove:
            S_curr.remove(x)
            in_set_curr[x] = False
        
        # Greedy refill with randomization using vectorized checks
        candidates = [x for x in range(1, N + 1) if not in_set_curr[x]]
        rng.shuffle(candidates)
        
        for x in candidates:
            if can_add_fast(x, in_set_curr):
                S_curr.add(x)
                in_set_curr[x] = True
        
        new_size = len(S_curr)
        delta = new_size - current_size
        
        # Acceptance criterion
        if delta > 0:
            # Always accept improvement
            current_set = S_curr
            current_in_set = in_set_curr
            current_size = new_size
            stuck_count = 0
            T = T_initial  # Reset temperature on improvement
        elif delta == 0:
            # Accept with probability 0.7
            if rng.random() < 0.7:
                current_set = S_curr
                current_in_set = in_set_curr
                stuck_count = max(0, stuck_count - 1)
            else:
                stuck_count += 1
        else:
            # Accept with probability exp(delta/T)
            if T > T_min and rng.random() < np.exp(delta / T):
                current_set = S_curr
                current_in_set = in_set_curr
                current_size = new_size
                stuck_count += 1
            else:
                stuck_count += 1
        
        # Update best
        if current_size > best_size:
            best_set = set(current_set)
            best_size = current_size
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If completely stuck, do a larger perturbation
        if stuck_count > 150:
            # Restart from best with larger perturbation
            current_set = set(best_set)
            current_in_set = set_to_np(current_set)
            current_size = best_size
            T = T_initial
            stuck_count = 0
    
    return sorted(best_set)
