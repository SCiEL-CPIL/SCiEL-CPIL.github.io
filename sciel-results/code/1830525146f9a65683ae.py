"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

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
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    base_rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree(seed_offset=0):
        """Greedy independent set with min-degree heuristic, randomized tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        local_rng = np.random.default_rng(42 + seed_offset)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            
            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower total degree
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                    else:
                        scores[idx] = 0
                # Add small random noise to break ties differently each restart
                noise = local_rng.random(len(cands)) * 0.1
                scores += noise
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0, seed_offset=0):
        """Enhanced simulated annealing with diverse moves and kick mechanism."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 4.0
        cooling = 0.9993
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        kick_counter = 0
        local_rng = np.random.default_rng(1000 + seed_offset)
        
        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            # Kick mechanism: if stuck too long, remove a random subset and re-add greedily
            if stuck_counter > 3000:
                # Remove 10-20% of in-set vertices randomly (more aggressive: up to 4)
                n_remove = max(1, min(4, int(0.15 * len(in_set_indices))))
                if n_remove < len(in_set_indices):
                    remove_idx = local_rng.choice(len(in_set_indices), size=n_remove, replace=False)
                    for ri in remove_idx:
                        v = in_set_indices[ri]
                        in_set[v] = False
                        count_conflicts[neighbors[v]] -= 1
                    current_size -= n_remove
                    stuck_counter = 0
                    kick_counter += 1
                    # Re-add greedily
                    for v in not_in_set:
                        if count_conflicts[v] == 0:
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            current_size += 1
                    not_in_set = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                    continue
            
            move_type = local_rng.random()
            
            if move_type < 0.35 and len(not_in_set) > 0:
                # ADD move: try to add a vertex with fewest conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[local_rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = local_rng.random() < 0.5
                else:
                    accept = local_rng.random() < np.exp(delta / T)
                
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
                    
            elif move_type < 0.55 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # 1-for-1 SWAP move: remove one in-set vertex, add one not-in-set vertex
                in_degrees = count_conflicts[in_set_indices]
                weights = 1.0 / (1.0 + in_degrees.astype(np.float64))
                weights /= weights.sum()
                v_remove = in_set_indices[local_rng.choice(len(in_set_indices), p=weights)]
                
                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[local_rng.choice(len(not_in_set), p=add_weights)]
                
                # Check if v_add conflicts with any in-set vertex (excluding v_remove)
                in_set_no_remove = in_set.copy()
                in_set_no_remove[v_remove] = False
                conflict_count = np.sum(conflicts[v_add] & in_set_no_remove)
                
                # Remove v_remove
                in_set[v_remove] = False
                count_conflicts[neighbors[v_remove]] -= 1
                current_size -= 1
                
                # Add v_add
                to_remove2 = np.where(conflicts[v_add] & in_set)[0]
                if len(to_remove2) > 0:
                    in_set[to_remove2] = False
                    for u in to_remove2:
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= len(to_remove2)
                
                in_set[v_add] = True
                count_conflicts[neighbors[v_add]] += 1
                current_size += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.75 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                # 2-for-1 SWAP move: remove two in-set vertices, add one not-in-set vertex
                # This helps escape local optima where single swaps don't help
                # Prefer removing vertices with high degree
                in_degrees = count_conflicts[in_set_indices]
                # Sample 2 vertices weighted by degree
                weights = 1.0 / (1.0 + in_degrees.astype(np.float64))
                weights /= weights.sum()
                
                # Ensure we don't pick the same vertex twice
                v_remove1_idx = local_rng.choice(len(in_set_indices), p=weights)
                v_remove1 = in_set_indices[v_remove1_idx]
                
                # Adjust weights for second pick
                weights2 = weights.copy()
                weights2[v_remove1_idx] = 0
                if weights2.sum() > 0:
                    weights2 /= weights2.sum()
                    v_remove2_idx = local_rng.choice(len(in_set_indices), p=weights2)
                else:
                    v_remove2_idx = (v_remove1_idx + 1) % len(in_set_indices)
                v_remove2 = in_set_indices[v_remove2_idx]
                
                # Check if v_remove1 and v_remove2 are the same (shouldn't be but safety)
                if v_remove1 == v_remove2:
                    stuck_counter += 1
                    continue
                
                # For v_add, prefer vertices with fewer conflicts
                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[local_rng.choice(len(not_in_set), p=add_weights)]
                
                # Remove both v_remove1 and v_remove2
                in_set[v_remove1] = False
                count_conflicts[neighbors[v_remove1]] -= 1
                current_size -= 1
                
                in_set[v_remove2] = False
                count_conflicts[neighbors[v_remove2]] -= 1
                current_size -= 1
                
                # Add v_add
                to_remove2 = np.where(conflicts[v_add] & in_set)[0]
                if len(to_remove2) > 0:
                    in_set[to_remove2] = False
                    for u in to_remove2:
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= len(to_remove2)
                
                in_set[v_add] = True
                count_conflicts[neighbors[v_add]] += 1
                current_size += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
            else:
                # Simple move: try to add a random non-conflicting vertex or skip
                if len(not_in_set) > 0:
                    v = not_in_set[local_rng.integers(len(not_in_set))]
                    if count_conflicts[v] == 0:
                        in_set[v] = True
                        count_conflicts[neighbors[v]] += 1
                        current_size += 1
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
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        seed_offset = n_restarts * 7
        
        # 1. Greedy Min-Degree Init with varied seed
        in_set = greedy_min_degree(seed_offset)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time, seed_offset=seed_offset)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish pass on best found so far if we improved
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time, seed_offset=seed_offset + 500)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final high-precision polish pass on the best solution found
    remaining_time = deadline - 3 - time.time()
    if remaining_time > 3 and best_size > 0:
        final_polish_time = min(remaining_time, 10.0)
        if final_polish_time > 2:
            final_result, final_size = sa_local_search(best_in_set, max_time=final_polish_time, seed_offset=9999)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
