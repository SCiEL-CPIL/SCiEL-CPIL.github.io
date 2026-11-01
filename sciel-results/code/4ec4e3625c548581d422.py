"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Support-level approach:
    # W4 vectors: weight 4, entries in {-1,0,1}, norm^2=4
    # Two W4 vectors conflict iff dot > 2
    # Within same support: max dot between distinct sign patterns = 2 (compatible!)
    # Between supports S1,S2: max dot = |S1∩S2|, conflict iff |S1∩S2| >= 3
    # So we need: choose supports with pairwise intersection <= 2
    # Each chosen support gives 16 mutually-compatible W4 vectors
    
    all_supports = list(itertools.combinations(range(D), 4))
    N_sup = len(all_supports)  # 330
    
    # Build conflict: supports i,j conflict if |Si ∩ Sj| >= 3
    # Precompute intersection sizes
    sup_arrays = [set(s) for s in all_supports]
    
    # Conflict matrix at support level
    sup_conflict = np.zeros((N_sup, N_sup), dtype=bool)
    for i in range(N_sup):
        for j in range(i+1, N_sup):
            inter = len(sup_arrays[i] & sup_arrays[j])
            if inter >= 3:
                sup_conflict[i, j] = True
                sup_conflict[j, i] = True
    
    # Find maximum independent set in this 330-node graph
    # Using greedy + local search
    
    neighbors = [np.where(sup_conflict[i])[0] for i in range(N_sup)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N_sup, dtype=bool)
    
    def greedy_min_degree():
        in_set = np.zeros(N_sup, dtype=bool)
        alive = np.ones(N_sup, dtype=bool)
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
                degrees[rem] -= sup_conflict[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        count_conflicts = np.zeros(N_sup, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(sup_conflict[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 1000:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Also try: add/remove swap moves
    def sa_with_swaps(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.999
        min_T = 0.05
        
        count_conflicts = np.zeros(N_sup, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.4:
                # Add move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(sup_conflict[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.7:
                # Remove move (perturbation)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                in_set[v] = False
                current_size -= 1
                count_conflicts[neighbors[v]] -= 1
                stuck_counter += 1
            else:
                # 2-for-1 swap: remove one in-set node, try to add two non-in-set nodes
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                # Try adding 2 random nodes from not_in_set
                if len(not_in_set) >= 2:
                    add_cands = not_in_set[rng.choice(len(not_in_set), size=2, replace=False)]
                    # Check if both can be added after removing v_remove
                    temp_conflict = sup_conflict.copy()
                    temp_conflict[v_remove, :] = False
                    temp_conflict[:, v_remove] = False
                    # Check if add_cands[0] and add_cands[1] are compatible with in_set minus v_remove and with each other
                    ok = True
                    for a in add_cands:
                        if sup_conflict[a, v_remove]:
                            continue  # removing v_remove helps
                        for b in in_set_indices:
                            if b == v_remove:
                                continue
                            if sup_conflict[a, b]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok and sup_conflict[add_cands[0], add_cands[1]]:
                        ok = False
                    if ok:
                        in_set[v_remove] = False
                        count_conflicts[neighbors[v_remove]] -= 1
                        for a in add_cands:
                            in_set[a] = True
                            count_conflicts[neighbors[a]] += 1
                        current_size += 1  # net +1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Use different seed per restart
        rng = np.random.default_rng(n_restarts * 7 + 13)
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search (choose method)
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        if rng.random() < 0.5:
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        else:
            sa_result, sa_size = sa_with_swaps(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish pass
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Build final vectors: for each chosen support, include all 16 sign patterns
    final_vectors_list = []
    for i in np.where(best_in_set)[0]:
        support = all_supports[i]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            final_vectors_list.append(v)
    
    result_w4 = np.array(final_vectors_list, dtype=np.int64)
    
    # Verify W4 vs W1 compatibility: W1 vector has entry ±2 at one position.
    # Dot with W4 vector: if W4 has ±1 at that position, dot = ±2. |dot| = 2 <= 2. OK!
    # W1 vs W1: same position: dot = 4 (conflict!), different positions: dot = 0.
    # So W1 vectors at same position conflict. We can only include one per position.
    # Wait: dot of (2,0,...) and (-2,0,...) = -4. dot of (2,0,...) and (2,0,...) = 4.
    # Conflict if dot > 2. So (2,...) and (2,...) conflict (dot=4), (2,...) and (-2,...) don't (dot=-4).
    # We can include both +2 and -2 at each position? No: dot=4 for same sign, -4 for opposite.
    # -4 is not > 2, so they're compatible! But wait, we need min pairwise distance >= max norm.
    # dist^2 = ||a||^2 + ||b||^2 - 2*a·b = 4 + 4 - 2*a·b = 8 - 2*a·b
    # Need dist^2 >= max_norm^2 = 4, so 8 - 2*a·b >= 4, i.e., a·b <= 2.
    # For W1 same position same sign: a·b = 4 > 2. CONFLICT.
    # For W1 same position opposite sign: a·b = -4 <= 2. OK.
    # So we can include both +2e_i and -2e_i for each i!
    
    final_vectors = np.vstack([result_w4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
