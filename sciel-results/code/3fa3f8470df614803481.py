import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all vectors with entries in {-2, -1, 0, 1, 2}
    # and squared norm exactly 4.
    # This gives us a large pool of candidates from which we can
    # find a large independent set (kissing configuration).
    
    # We'll generate vectors by weight (number of non-zero entries):
    # Weight 1: (±2) in one position -> norm²=4
    # Weight 2: (±1,±1) in two positions -> norm²=2, or (±2) in one -> already counted
    #           Actually weight 2 with norm²=4: need a²+b²=4 with a,b ∈ {-2,-1,0,1,2}
    #           (±2, 0) - weight 1
    #           (±1, ±1) - norm²=2, not 4
    #           So no weight 2 with norm²=4
    # Weight 3: a²+b²+c²=4, entries in {-2,-1,0,1,2}
    #           (±2, ±1, 0) permutations -> but that's weight 2
    #           (±1, ±1, ±1) -> norm²=3
    #           (±2, 0, 0) -> weight 1
    #           (±1, ±1, 0) -> norm²=2
    #           So no weight 3 with norm²=4
    # Weight 4: a²+b²+c²+d²=4, entries in {-1,0,1} (can't have 2 since 4+...>4)
    #           (±1, ±1, ±1, ±1) -> norm²=4 ✓
    # Weight 5+: sum of squares = 4 with 5+ non-zero entries impossible (min is 5)
    
    # So candidates are:
    # Type A: weight 1, entry ±2
    # Type B: weight 4, entries ±1
    
    # But wait, we also need to consider that the problem says "nonzero integer vectors"
    # and "minimum pairwise distance is at least the maximum vector norm."
    # If all vectors have the same norm, then max norm = that norm.
    # Distance between u and v: ||u-v||² = ||u||² + ||v||² - 2u·v = 2||u||² - 2u·v (if ||u||=||v||)
    # We need ||u-v|| >= ||u||, so ||u-v||² >= ||u||²
    # 2||u||² - 2u·v >= ||u||²
    # ||u||² >= 2u·v
    # For norm²=4: 4 >= 2u·v, i.e., u·v <= 2
    
    # So we need to find a large set of vectors from our candidate pool
    # where every pair has dot product <= 2.
    
    # Let's generate all candidates:
    candidates = []
    
    # Type A: weight 1, ±2
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    
    # Type B: weight 4, ±1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    cand_array = np.array(candidates, dtype=np.int64)
    N = len(cand_array)
    
    if N == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Compute dot product matrix
    dots = cand_array @ cand_array.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        return in_set
    
    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        not_in_set = np.where(~in_set)[0]
        
        step = 0
        while time.time() < sa_deadline:
            step += 1
            if len(not_in_set) == 0:
                break
            
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            if min_count == 0:
                zero_cands = not_in_set[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]
                
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                
                mask = not_in_set != v
                not_in_set = not_in_set[mask]
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue
            
            if step % 5 == 0 and current_size >= 2:
                in_set_idx = np.where(in_set)[0]
                if len(in_set_idx) >= 2:
                    i1, i2 = rng.choice(in_set_idx, size=2, replace=False)
                    
                    in_set[i1] = False
                    in_set[i2] = False
                    current_size -= 2
                    count_conflicts[neighbors[i1]] -= 1
                    count_conflicts[neighbors[i2]] -= 1
                    
                    temp_not_in = np.concatenate([not_in_set, np.array([i1, i2])])
                    counts_now = count_conflicts[temp_not_in]
                    addable = temp_not_in[counts_now == 0]
                    
                    if len(addable) >= 1:
                        if len(addable) >= 2 and rng.random() < 0.3:
                            v1, v2 = rng.choice(addable, size=2, replace=False)
                            in_set[v1] = True
                            in_set[v2] = True
                            current_size += 2
                            count_conflicts[neighbors[v1]] += 1
                            count_conflicts[neighbors[v2]] += 1
                        else:
                            v = addable[rng.integers(len(addable))]
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                        
                        mask = np.isin(not_in_set, [i1, i2], invert=True)
                        not_in_set = not_in_set[mask]
                        new_items = [x for x in [i1, i2] if not in_set[x]]
                        if new_items:
                            not_in_set = np.concatenate([not_in_set, np.array(new_items)])
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        continue
                    else:
                        in_set[i1] = True
                        in_set[i2] = True
                        current_size += 2
                        count_conflicts[neighbors[i1]] += 1
                        count_conflicts[neighbors[i2]] += 1
                    continue
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
            
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if len(to_remove) > 0:
                    mask = not_in_set != v
                    not_in_set = np.concatenate([not_in_set[mask], to_remove])
                else:
                    mask = not_in_set != v
                    not_in_set = not_in_set[mask]
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    def exact_polish(in_set_init, max_time=5.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                if len(zero_cands) > 0:
                    for v in zero_cands:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
            
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1
                
                i_neighbors = neighbors[i]
                cand_mask = ~in_set[i_neighbors]
                if np.any(cand_mask):
                    cand_indices = i_neighbors[cand_mask]
                    cand_counts = count_conflicts[cand_indices]
                    addable = cand_indices[cand_counts == 0]
                    if len(addable) > 0:
                        best_v = addable[0]
                        best_score = -1
                        for v in addable:
                            v_nbrs = neighbors[v]
                            score = int(np.sum(~in_set[v_nbrs]))
                            if score > best_score:
                                best_score = score
                                best_v = v
                        v = best_v
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        continue
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors[i]] += 1
        
        return local_best_in_set, local_best_size
    
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        if n_restarts % 3 == 0:
            in_set = greedy_min_degree()
        elif n_restarts % 3 == 1:
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            degrees = neighbor_counts.copy()
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                top_k = min(3, len(cands))
                chosen = cands[rng.integers(top_k)]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        else:
            rng_local = np.random.default_rng(n_restarts * 12345)
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            degrees = neighbor_counts.copy()
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                chosen = cands[rng_local.integers(len(cands))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
        
        sa_time = min(remaining_time * 0.45, 12.0)
        if sa_time < 2:
            sa_time = 2.0
        
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 6.0)
            if polish_time > 1:
                polish_result, polish_size = exact_polish(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        for pass_num in range(3):
            remaining_time = deadline - 1 - pass_num - time.time()
            if remaining_time < 0.5:
                break
            polish_time = min(remaining_time * 0.6, 4.0)
            if polish_time < 0.5:
                break
            polish_result, polish_size = exact_polish(best_in_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
            else:
                remaining_time = deadline - 1 - time.time()
                if remaining_time > 1:
                    sa_time = min(remaining_time * 0.5, 2.0)
                    if sa_time > 0.5:
                        sa_result, sa_size = local_search(
                            best_in_set, max_time=sa_time, min_T=0.01, cooling=0.9999
                        )
                        if sa_size > best_size:
                            best_size = sa_size
                            best_in_set = sa_result.copy()
                        break
    
    final_indices = np.where(best_in_set)[0]
    return cand_array[final_indices]

# EVOLVE-BLOCK-END
