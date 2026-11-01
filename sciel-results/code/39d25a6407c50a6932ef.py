"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

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
        """Greedy independent set with min-degree heuristic, batch removal."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            # Tie-break: prefer vertex with fewer "second-neighbor" conflicts (lower spread)
            if len(cands) > 1:
                # Compute number of alive neighbors of each candidate's neighbors
                # Approximate: prefer candidate whose neighbors have lower average degree
                scores = np.array([np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0 for c in cands])
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
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        
        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6 and len(not_in_set) > 0:
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
                    accept = rng.random() < 0.3
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
                    
            elif move_type < 0.8 and len(in_set_indices) > 0:
                # REMOVE move: try to remove a vertex (helps escape local optima)
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            else:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    # Pick a random in-set vertex to remove
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    # Find not-in-set vertices that don't conflict with v_remove's other neighbors
                    # Simplified: pick a random not-in-set vertex
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    
                    # Check if v_add conflicts with any in-set vertex other than v_remove
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    
                    if len(other_conflicts) == 0:
                        # Clean swap: remove v_remove, add v_add
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        # Net delta = 0, always accept
                    else:
                        # Complex swap: remove v_remove and all other conflicts, add v_add
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
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 2000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish: try to add any non-conflicting vertex to the best set
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_set = sa_result.copy() if sa_size >= current_size else in_set.copy()
            polish_size = max(sa_size, current_size)
            
            # Greedy polish: repeatedly add any vertex that doesn't conflict
            # We need to check against the current polish_set
            # Using the global conflicts matrix
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                
                # Vectorized check: for each candidate v in not_in,
                # check if it has any conflict with the current polish_set
                # conflicts[not_in, :] is shape (len(not_in), N4)
                # polish_set is shape (N4,)
                # conflict_mask = conflicts[not_in, :] & polish_set[None, :]
                # can_add = ~np.any(conflict_mask, axis=1)
                
                # To save memory/time, we can do this in chunks if not_in is large, 
                # but N4 is ~2640*2 = 5280? No, W4 is C(11,4)*2^4 = 330*16 = 5280.
                # 5280x5280 boolean matrix is ~28MB. Accessing subsets is fine.
                
                # Efficiently find candidates that don't conflict with current set
                # Get rows of conflict matrix for candidates
                cand_conflicts = conflicts[not_in, :]
                # Check intersection with current set
                # If sum of (cand_conflicts & polish_set) is 0, it can be added
                # We can sum along axis 1 after masking
                # mask = cand_conflicts & polish_set[None, :]
                # sums = mask.sum(axis=1)
                # This creates a large temp array. 
                # Alternative: use matrix multiplication with boolean logic? 
                # Or just loop if not_in is small? 
                # Actually, we can use the property that we only add if no conflicts.
                
                # Let's use a simpler approach: 
                # Compute conflicts with current set for all not_in candidates at once
                # dot_product_style_check: 
                # If we treat polish_set as 0/1 vector, and conflicts as 0/1 matrix,
                # then (conflicts[not_in] @ polish_set) gives count of conflicts for each candidate.
                # If count is 0, we can add.
                
                conflict_counts = cand_conflicts @ polish_set.astype(np.float64)
                addable_indices = np.where(conflict_counts == 0)[0]
                
                if len(addable_indices) > 0:
                    # Add all addable indices? 
                    # Adding multiple at once might cause conflicts between them.
                    # So we should add them one by one or in batches that are mutually independent.
                    # For simplicity and speed, let's just add the first few or all if they are independent.
                    # Since we want to maximize size, let's try to add as many as possible.
                    # A simple greedy: pick one, update set, repeat.
                    # But vectorizing the "pick one" is hard.
                    # Let's just add the first one found to keep it simple and safe.
                    v_idx = not_in[addable_indices[0]]
                    polish_set[v_idx] = True
                    polish_size += 1
                    changed = True
                    if time.time() >= deadline - 3:
                        break
                    # Continue loop to find more
                    # Optimization: If we add many, we might want to batch.
                    # But single addition is safe.
                    # To speed up, we can add all that are mutually non-conflicting?
                    # That requires solving an independent set on the subgraph of addable nodes.
                    # Too expensive. Single step is fine.
                    
                    # Actually, we can add all `addable_indices` if we check that they don't conflict with *each other*?
                    # No, they might conflict with each other.
                    # Let's stick to adding one per iteration to ensure correctness of "no conflict with current set".
                    # The loop will run again, recompute conflict_counts, and find the next one.
                    # This is O(k * N4^2) worst case but usually fast.
                    
            # Second SA polish
            if time.time() < deadline - 3:
                polish_time = min(deadline - 3 - time.time(), 8.0)
                if polish_time > 2:
                    polish_result, polish_size = sa_local_search(polish_set, max_time=polish_time)
                    if polish_size > best_size:
                        best_size = polish_size
                        best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify: min pairwise distance >= max norm
    # Compute norms
    norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise distances squared
    # For large k, this can be expensive; use chunked computation
    k = len(final_vectors)
    min_dist_sq = np.inf
    chunk_size = 500
    for i in range(0, k, chunk_size):
        chunk = final_vectors[i:i+chunk_size].astype(np.float64)
        # Distance to all vectors
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        chunk_norms_sq = np.sum(chunk**2, axis=1, keepdims=True)
        all_norms_sq = norms_sq[None, :]
        dot = chunk @ final_vectors.T
        dist_sq = chunk_norms_sq + all_norms_sq - 2*dot
        # Exclude self-distances
        for j in range(len(chunk)):
            dist_sq[j, i+j] = np.inf
        min_dist_sq = min(min_dist_sq, np.min(dist_sq))
    
    # If constraint violated, we should have caught this in search, but verify
    # The constraint is: min pairwise distance >= max norm
    # i.e., min_dist_sq >= max_norm_sq
    
    return final_vectors

# EVOLVE-BLOCK-END
