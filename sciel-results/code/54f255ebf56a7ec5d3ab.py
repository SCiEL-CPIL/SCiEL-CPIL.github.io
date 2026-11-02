"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
            
    vectors_v4 = np.array(v4_list, dtype=np.int64)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as sorted arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_init(seed_rng=None):
        """Greedy independent set with min-degree heuristic and random tie-breaking."""
        if seed_rng is None:
            seed_rng = rng
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 8:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            
            chosen = cands[seed_rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.concatenate(([chosen], nbrs))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized degree update
                for nbr in nbrs:
                    if alive[nbr] == False:
                        # Already removed, but need to update degrees of remaining
                        pass
                # More efficient: subtract conflicts with chosen from remaining
                degrees[rem] -= conflicts[rem, chosen]
                
        return in_set
    
    def polish_add(in_set, max_time=3.0):
        """Add all possible non-conflicting vertices in batches."""
        polish = in_set.copy()
        polish_size = int(polish.sum())
        polish_deadline = time.time() + max_time
        
        changed = True
        while changed and time.time() < polish_deadline:
            changed = False
            not_in = np.where(~polish)[0]
            if len(not_in) == 0:
                break
            
            # Find vertices with zero conflicts with current set
            conflict_counts = conflicts[not_in] @ polish.astype(np.float64)
            addable = not_in[conflict_counts == 0]
            
            if len(addable) == 0:
                break
            
            # Greedily add non-conflicting vertices from addable
            # Sort by degree for better packing
            addable_degrees = neighbor_counts[addable]
            order = np.argsort(addable_degrees)
            sorted_addable = addable[order]
            
            added_this_round = []
            for v in sorted_addable:
                if time.time() >= polish_deadline:
                    break
                if len(added_this_round) == 0:
                    conflict_with_added = False
                else:
                    conflict_with_added = np.any(conflicts[v, added_this_round])
                
                if not conflict_with_added:
                    polish[v] = True
                    added_this_round.append(v)
                    polish_size += 1
                    changed = True
            
            if len(added_this_round) == 0:
                break
        
        return polish, polish_size
    
    def sa_search(in_set_init, max_time=8.0):
        """SA with add, remove, and swap moves using incremental conflict counts."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 8.0
        cooling = 0.9998
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Precompute for speed
        conflicts_bool = conflicts  # already boolean
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.50 and len(not_in_set) > 0:
                # ADD move: try to add a vertex
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.2
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts_bool[v] & in_set)[0]
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
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                counts_not_in = count_conflicts[not_in_set]
                min_c = np.min(counts_not_in)
                min_cands_not_in = not_in_set[counts_not_in == min_c]
                v_add = min_cands_not_in[rng.integers(len(min_cands_not_in))]
                
                # Check if v_add conflicts with any in-set vertex other than v_remove
                conflicts_with = np.where(conflicts_bool[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                else:
                    # Complex swap
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
            else:
                # 2-SWAP move: remove 2 in-set, add 2 not-in-set
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v_remove1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v_remove2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v_remove1 == v_remove2:
                        v_remove2 = in_set_indices[(np.where(in_set_indices == v_remove1)[0][0] + 1) % len(in_set_indices)]
                    
                    counts_not_in = count_conflicts[not_in_set]
                    min_c = np.min(counts_not_in)
                    min_cands_not_in = not_in_set[counts_not_in == min_c]
                    v_add1 = min_cands_not_in[rng.integers(len(min_cands_not_in))]
                    v_add2 = min_cands_not_in[rng.integers(len(min_cands_not_in))]
                    if v_add1 == v_add2:
                        v_add2 = min_cands_not_in[(np.where(min_cands_not_in == v_add1)[0][0] + 1) % len(min_cands_not_in)]
                    
                    # Check conflicts
                    # v_add1 and v_add2 must not conflict with each other
                    if conflicts_bool[v_add1, v_add2]:
                        stuck_counter += 1
                    else:
                        # Check conflicts with remaining in_set (after removing v_remove1, v_remove2)
                        in_set_temp = in_set.copy()
                        in_set_temp[v_remove1] = False
                        in_set_temp[v_remove2] = False
                        
                        conflicts_v1 = np.where(conflicts_bool[v_add1] & in_set_temp)[0]
                        conflicts_v2 = np.where(conflicts_bool[v_add2] & in_set_temp)[0]
                        all_conflicts = np.unique(np.concatenate((conflicts_v1, conflicts_v2))) if (len(conflicts_v1) > 0 or len(conflicts_v2) > 0) else np.array([], dtype=np.int64)
                        
                        to_remove = np.unique(np.concatenate(([v_remove1, v_remove2], all_conflicts)))
                        delta = 2 - len(to_remove)
                        
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add1] = True
                            in_set[v_add2] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add1]] += 1
                            count_conflicts[neighbors[v_add2]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 2000:
                T = max(T, 5.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 300
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy init with varied random seed
        restart_rng = np.random.default_rng(42 + n_restarts * 7919)
        in_set = greedy_init(restart_rng)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Polish: quick greedy add
        remaining = deadline - 10 - time.time()
        if remaining > 3:
            polished, polished_size = polish_add(in_set, max_time=min(remaining * 0.12, 2.5))
            if polished_size > current_size:
                in_set = polished
                current_size = polished_size
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
        
        # 3. SA Local Search
        remaining = deadline - 10 - time.time()
        if remaining > 3:
            sa_time = min(remaining * 0.65, 10.0)
            if sa_time < 1.0:
                sa_time = 1.0
                
            sa_result, sa_size = sa_search(in_set, max_time=sa_time)
            
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()
                in_set = sa_result
                current_size = sa_size
        
        # 4. Final polish after SA
        remaining = deadline - 6 - time.time()
        if remaining > 2 and current_size == best_size:
            polished, polished_size = polish_add(in_set, max_time=min(remaining * 0.4, 3.0))
            if polished_size > best_size:
                best_size = polished_size
                best_in_set = polished.copy()
        
        # Adaptive: if we've been stuck, do more aggressive restarts
        if n_restarts > 30:
            elapsed = time.time() - start_time
            if elapsed > 80 and best_size < 200:
                # Try a different strategy: random restart with larger SA
                pass
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify: min pairwise distance >= max norm
    norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    k = len(final_vectors)
    min_dist_sq = np.inf
    chunk_size = 500
    for i in range(0, k, chunk_size):
        chunk = final_vectors[i:i+chunk_size].astype(np.float64)
        chunk_norms_sq = np.sum(chunk**2, axis=1, keepdims=True)
        all_norms_sq = norms_sq[None, :]
        dot = chunk @ final_vectors.T
        dist_sq = chunk_norms_sq + all_norms_sq - 2*dot
        for j in range(len(chunk)):
            dist_sq[j, i+j] = np.inf
        chunk_min = np.min(dist_sq)
        if chunk_min < min_dist_sq:
            min_dist_sq = chunk_min
    
    return final_vectors

# EVOLVE-BLOCK-END
