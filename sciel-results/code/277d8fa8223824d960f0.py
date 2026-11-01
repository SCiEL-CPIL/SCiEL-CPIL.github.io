"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(777)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_init(seed):
        """Fast greedy independent set using min-degree heuristic."""
        local_rng = np.random.default_rng(seed)
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 3:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            degs = degrees[alive_indices]
            min_deg = np.min(degs)
            cands = alive_indices[degs == min_deg]
            chosen = cands[local_rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def greedy_polish(in_set_init):
        """Add any non-conflicting vertices iteratively."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            # Find vertices with no conflicts with current set
            conflict_counts = np.array([
                int((conflicts[v] & in_set).sum()) for v in not_in
            ])
            addable = not_in[conflict_counts == 0]
            
            if len(addable) > 0:
                # Greedily add mutually non-conflicting vertices
                remaining = addable.copy()
                while len(remaining) > 0:
                    v = remaining[0]
                    in_set[v] = True
                    current_size += 1
                    changed = True
                    # Remove v and its neighbors from remaining
                    nbrs_set = set(neighbors[v])
                    nbrs_set.add(v)
                    remaining = np.array([x for x in remaining if x not in nbrs_set])
        
        return in_set, current_size
    
    def sa_local_search(in_set_init, max_time=5.0, T_start=2.0):
        """Simulated annealing with add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = T_start
        cooling = 0.9998
        min_T = 0.05
        
        # Precompute conflict counts
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55:
                # ADD move: pick vertex with fewest conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
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
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.90 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Find swap candidates with few conflicts other than v_remove
                n_sample = min(20, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_cands = not_in_set[sample_idx]
                
                best_swap = None
                best_delta = -100
                for v_add in sample_cands:
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    d = 1 - 1 - len(other_conflicts)
                    if d > best_delta:
                        best_delta = d
                        best_swap = (v_remove, v_add, other_conflicts)
                
                if best_swap is not None and best_delta >= -3:
                    v_rm, v_ad, others = best_swap
                    if best_delta >= 0 or rng.random() < np.exp(best_delta / T):
                        to_remove = np.unique(np.concatenate(([v_rm], others))) if len(others) > 0 else np.array([v_rm])
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_ad] = True
                        current_size += best_delta
                        count_conflicts[neighbors[v_ad]] += 1
                else:
                    stuck_counter += 1
                    
            else:
                # 2-SWAP: remove 2, add 2
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 == v2:
                        stuck_counter += 1
                    else:
                        n_sample = min(10, len(not_in_set))
                        sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                        sample_cands = not_in_set[sample_idx]
                        
                        found = []
                        for v_add in sample_cands:
                            conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                            other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                            if len(other_conflicts) == 0:
                                found.append(v_add)
                                if len(found) == 2:
                                    break
                        
                        if len(found) == 2:
                            in_set[v1] = False
                            in_set[v2] = False
                            count_conflicts[neighbors[v1]] -= 1
                            count_conflicts[neighbors[v2]] -= 1
                            for v_ad in found:
                                in_set[v_ad] = True
                                count_conflicts[neighbors[v_ad]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Luby sequence for restart scheduling
    luby_pattern = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    
    n_restarts = 0
    luby_idx = 0
    
    while time.time() < deadline - 5 and n_restarts < 200:
        n_restarts += 1
        
        luby_len = luby_pattern[luby_idx % len(luby_pattern)]
        luby_idx += 1
        
        # 1. Greedy Init
        seed = n_restarts * 31 + 17
        in_set = greedy_init(seed)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 3:
            break
        
        base_sa_time = min(remaining_time * 0.35, 6.0)
        sa_time = base_sa_time * (luby_len / 4.0)
        sa_time = max(sa_time, 0.8)
        sa_time = min(sa_time, remaining_time * 0.5)
        
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Greedy polish
        remaining_time = deadline - 3 - time.time()
        if remaining_time > 1.5:
            polish_set, polish_size = greedy_polish(sa_result if sa_size >= current_size else in_set)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_set.copy()
        
        # 4. Occasional second SA on best
        remaining_time = deadline - 2 - time.time()
        if remaining_time > 4 and n_restarts % 7 == 0:
            polish_time = min(remaining_time * 0.4, 4.0)
            if polish_time > 1.5:
                polish_result, polish_size = sa_local_search(best_in_set, max_time=polish_time, T_start=1.0)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final greedy polish
    if time.time() < deadline - 1:
        final_polish, final_size = greedy_polish(best_in_set)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
