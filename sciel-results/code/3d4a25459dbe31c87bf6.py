import numpy as np
import itertools
import time

D = 11

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
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
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
                # Update degrees: for each remaining node, subtract conflicts with removed nodes
                for r in rem:
                    deg_change = conflicts[r, to_remove].sum()
                    degrees[r] -= deg_change
                    
        return in_set
    
    def advanced_local_search(in_set_init, max_time=15.0):
        """Local search with add, remove, swap, and 2-for-1 moves with adaptive perturbation."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        # Track conflicts: for each node not in set, how many neighbors are in set
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        sa_deadline = time.time() + max_time
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        stuck_counter = 0
        perturbation_counter = 0
        
        while time.time() < sa_deadline:
            # Choose move type
            move_type = rng.integers(4)  # 0: add, 1: remove, 2: swap, 3: 2-for-1
            
            accepted = False
            
            if move_type == 0:  # Add a vertex
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                # Prefer vertices with fewest conflicts
                counts = conflict_count[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = conflict_count[v]
                delta = 1 - k
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Remove conflicting vertices
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    for u in to_remove:
                        in_set[u] = False
                        conflict_count[neighbors[u]] -= 1
                        # Update conflict_count for v's other neighbors
                    in_set[v] = True
                    current_size += delta
                    # Update conflict counts
                    for u in neighbors[v]:
                        if not in_set[u]:
                            conflict_count[u] += 1
                    accepted = True
                    
            elif move_type == 1:  # Remove a vertex (to enable future adds)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                # Removing v: delta = -1 + (number of vertices that were blocked by v and can now be added)
                # Simplified: just remove, delta = -1
                delta = -1
                if rng.random() < np.exp(delta / T):
                    in_set[v] = False
                    current_size += delta
                    for u in neighbors[v]:
                        if not in_set[u]:
                            conflict_count[u] -= 1
                    accepted = True
                    
            elif move_type == 2:  # Swap: remove v from set, add u not in set
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                # u should not conflict with (in_set \ {v}) 
                # Find u not in set, not neighbor of v, with minimal conflicts with in_set\{v}
                # Simplified: pick random u not in set
                u = not_in_set[rng.integers(len(not_in_set))]
                if u == v or conflicts[v, u]:
                    continue
                # Check if u conflicts with other in_set members
                u_conflicts = np.where(conflicts[u] & in_set)[0]
                u_conflicts = u_conflicts[u_conflicts != v]
                k = len(u_conflicts)
                delta = 1 - k  # Remove v, add u, but u may conflict with k others
                # Actually delta = -1 (remove v) + 1 (add u) - k (remove k conflicting) = -k
                # Wait: we remove v, add u. If u conflicts with k in_set members (excluding v), we need to remove those too.
                # So net change = -1 (remove v) + 1 (add u) - k (remove k) = -k
                if k == 0:
                    delta = 0
                elif rng.random() < np.exp(-k / T):
                    # Execute swap with removals
                    for w in u_conflicts:
                        in_set[w] = False
                        conflict_count[neighbors[w]] -= 1
                    in_set[v] = False
                    for w in neighbors[v]:
                        if not in_set[w]:
                            conflict_count[w] -= 1
                    in_set[u] = True
                    for w in neighbors[u]:
                        if not in_set[w]:
                            conflict_count[w] += 1
                    current_size += (-k)
                    accepted = True
                    
            else:  # 2-for-1: remove two vertices, add one that was blocked
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) < 2 or len(not_in_set) == 0:
                    continue
                # Pick a candidate u not in set
                u = not_in_set[rng.integers(len(not_in_set))]
                u_conflicts = np.where(conflicts[u] & in_set)[0]
                if len(u_conflicts) > 2:
                    continue
                if len(u_conflicts) == 0:
                    continue  # Should have been added already
                # Remove u_conflicts, add u
                k = len(u_conflicts)
                delta = 1 - k
                if delta > 0 or rng.random() < np.exp(delta / T):
                    for w in u_conflicts:
                        in_set[w] = False
                        conflict_count[neighbors[w]] -= 1
                    in_set[u] = True
                    for w in neighbors[u]:
                        if not in_set[w]:
                            conflict_count[w] += 1
                    current_size += delta
                    accepted = True
            
            if accepted:
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            # Adaptive perturbation
            if stuck_counter > 2000:
                # Perturb: randomly flip some vertices
                n_flips = rng.integers(5, 15)
                flip_indices = rng.choice(N4, size=min(n_flips, N4), replace=False)
                for idx in flip_indices:
                    if in_set[idx]:
                        # Remove
                        in_set[idx] = False
                        for w in neighbors[idx]:
                            if not in_set[w]:
                                conflict_count[w] -= 1
                        current_size -= 1
                    else:
                        # Try to add
                        k = conflict_count[idx]
                        if k == 0:
                            in_set[idx] = True
                            for w in neighbors[idx]:
                                if not in_set[w]:
                                    conflict_count[w] += 1
                            current_size += 1
                stuck_counter = 0
                perturbation_counter += 1
                T = max(T, 3.0)
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Advanced Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        ls_time = min(remaining_time * 0.5, 15.0)
        if ls_time < 2:
            ls_time = 2.0
            
        ls_result, ls_size = advanced_local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # 3. Polish pass
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and ls_size > current_size:
            polish_time = min(remaining_time * 0.4, 10.0)
            if polish_time > 2:
                polish_result, polish_size = advanced_local_search(ls_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
