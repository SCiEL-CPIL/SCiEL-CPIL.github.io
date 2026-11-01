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
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
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
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict SA with ADD and SWAP moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.999
        min_T = 0.02
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type: ADD (70%) or SWAP (30%)
            use_swap = rng.random() < 0.3
            
            if use_swap and current_size > 0:
                # SWAP: remove one in-set vertex, add one not-in-set vertex
                in_set_indices = np.where(in_set)[0]
                not_in_set_indices = np.where(~in_set)[0]
                
                if len(in_set_indices) == 0 or len(not_in_set_indices) == 0:
                    continue
                
                # Pick vertex to remove (prefer high conflict count)
                rem_candidates = in_set_indices
                # Weight by how many conflicts it has in the set
                rem_conflicts_in_set = np.array([
                    int(np.sum(conflicts[rem_candidates[j]] & in_set)) 
                    for j in range(len(rem_candidates))
                ])
                # Normalize to probability
                rem_total = rem_conflicts_in_set.sum()
                if rem_total == 0:
                    rem_prob = np.ones(len(rem_candidates)) / len(rem_candidates)
                else:
                    rem_prob = rem_conflicts_in_set / rem_total
                rem_idx = rng.choice(len(rem_candidates), p=rem_prob)
                v_remove = rem_candidates[rem_idx]
                
                # Pick vertex to add (prefer low conflict count)
                counts = count_conflicts[not_in_set_indices]
                min_count = np.min(counts)
                min_cands = not_in_set_indices[counts == min_count]
                v_add = min_cands[rng.integers(len(min_cands))]
                
                # Check if v_add conflicts with v_remove's other neighbors
                # After removing v_remove, v_add's conflicts are reduced by 1 if v_remove was a conflict
                k_add = int(count_conflicts[v_add])
                if conflicts[v_add, v_remove]:
                    k_add -= 1
                
                delta = 1 - k_add  # net change: add 1, remove k_add conflicts
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    # Execute swap
                    in_set[v_remove] = False
                    current_size -= 1
                    # Update count_conflicts for v_remove's neighbors
                    for u in neighbors[v_remove]:
                        count_conflicts[u] -= 1
                    
                    # Now add v_add
                    to_remove = np.where(conflicts[v_add] & in_set)[0]
                    for u in to_remove:
                        in_set[u] = False
                        current_size -= 1
                        for w in neighbors[u]:
                            count_conflicts[w] -= 1
                    
                    in_set[v_add] = True
                    current_size += 1
                    for u in neighbors[v_add]:
                        count_conflicts[u] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # ADD move: add a vertex not in set
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
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200
    seed_offset = 0
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        seed_offset += 1
        rng = np.random.default_rng(42 + seed_offset * 7919)
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        # Adaptive SA time: more time for early restarts, less for later
        progress = n_restarts / max_restarts
        sa_time = min(remaining_time * (0.5 + 0.3 * (1 - progress)), 12.0)
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish pass on best found
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and sa_size >= best_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
