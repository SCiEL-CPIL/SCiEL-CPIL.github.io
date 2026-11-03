import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate Candidate Vectors
    # We focus on squared norms 1, 2, 4 to keep M=4 fixed and graph manageable.
    candidates = []
    
    # Type 1: Norm^2 = 1 (single 1)
    # 11 * 2 = 22
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Type 2: Norm^2 = 2 (two 1s)
    # C(11,2) * 4 = 55 * 4 = 220
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Norm^2 = 4
    # Subtype 3a: (2, 0, ...) -> 11 * 2 = 22
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Subtype 3b: (1, 1, 1, 1, 0, ...) -> C(11,4) * 16 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # 2. Precompute Dot Products and Norms
    # M = candidates @ candidates.T
    # To save memory/time, we can compute in chunks if N is large, but 5500 is fine.
    M = candidates @ candidates.T
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # 3. Build Conflict Matrix
    # Condition for conflict (violation of distance >= M_max):
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We require dist^2 >= M_max.
    # Let M_max = 4 (since all candidates have norm^2 <= 4).
    # So: ||u||^2 + ||v||^2 - 2 u.v >= 4
    # 2 u.v <= ||u||^2 + ||v||^2 - 4
    # Conflict if 2 u.v > ||u||^2 + ||v||^2 - 4
    
    M_max = 4
    # norms_sq_matrix[i,j] = norms_sq[i] + norms_sq[j]
    norms_sq_matrix = norms_sq[:, None] + norms_sq[None, :]
    
    # Conflict if 2*M[i,j] > norms_sq_matrix[i,j] - M_max
    # Note: M[i,i] = norms_sq[i]. 2*norms_sq[i] > norms_sq[i] + norms_sq[i] - 4 => 0 > -4 (False). Diagonal is 0.
    Conflict = (2 * M > (norms_sq_matrix - M_max)).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Convert to boolean for faster logical ops
    Conflict_bool = Conflict.astype(bool)
    
    # 4. Search for Max Independent Set
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
        
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        # Degree in the induced subgraph on 'active'
        # Initial degree is total degree
        deg = Conflict_bool.sum(axis=1).astype(np.int64)
        
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Find min degree among active
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            
            # Pick one of the min-degree vertices (random tie-break)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors from active
            # Neighbors in active: Conflict_bool[idx, :] & active
            neighbors_mask = Conflict_bool[idx, :].astype(bool)
            remove_mask = np.zeros(N, dtype=bool)
            remove_mask[idx] = True
            remove_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(remove_mask)[0]
            
            # Update degrees for remaining active vertices
            # deg[v] -= count of removed neighbors
            if len(removed_indices) > 0:
                # Sum conflicts of removed vertices
                # conflict_counts[v] = sum(Conflict_bool[removed_indices, v])
                conflict_counts = Conflict_bool[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= conflict_counts[active]
            
            active[remove_mask] = False
            
        return selected

    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
        
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set), dtype=np.int64)
        
        improved = True
        max_iters = 200
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Shuffle order to avoid bias
            rng_ls = np.random.default_rng()
            order = rng_ls.permutation(len(selected_arr))
            selected_arr = selected_arr[order]
            
            for i_idx in range(len(selected_arr)):
                if time.time() > deadline:
                    return list(selected_set)
                
                u = selected_arr[i_idx]
                
                # 1. Remove u from selected_set
                selected_set.remove(u)
                S_mask = np.zeros(N, dtype=bool)
                S_mask[list(selected_set)] = True
                
                # 2. Find candidates v not in S that are compatible with S
                # Compatible if no conflict with any in S
                # conflict_sum = Conflict_bool[S_mask, :].sum(axis=0)
                # candidates_v = np.where((~S_mask) & (conflict_sum == 0))[0]
                
                # Optimization: Use matrix multiplication or dot product?
                # Conflict_bool is uint8 or bool.
                # Let's use sum.
                if len(selected_set) > 0:
                    conflict_counts = Conflict_bool[S_mask, :].sum(axis=0)
                else:
                    conflict_counts = np.zeros(N, dtype=np.int64)
                    
                valid_v = np.where((~S_mask) & (conflict_counts == 0))[0]
                
                if len(valid_v) < 2:
                    # Cannot add 2, put u back
                    selected_set.add(u)
                    continue
                
                # 3. Check if any pair in valid_v is mutually compatible
                # We need v1, v2 in valid_v such that Conflict_bool[v1, v2] == 0
                # This is finding an edge in the complement graph of Conflict induced on valid_v
                # Or simply: does valid_v contain an independent set of size 2?
                # Since Conflict_bool is symmetric, we can check:
                # For each v1 in valid_v, check if any v2 in valid_v (v2 != v1) has Conflict_bool[v1, v2] == 0
                
                # To speed this up, we can just check the first few?
                # Or vectorize:
                # Sub_matrix = Conflict_bool[np.ix_(valid_v, valid_v)]
                # If any 0 in off-diagonal, we have a pair.
                
                if len(valid_v) > 100:
                    # Sample to keep it fast? No, N is small enough.
                    # But constructing submatrix is O(K^2).
                    # Let's just check if the minimum degree in the induced conflict graph on valid_v is < K-1
                    # Actually, simpler:
                    # Check if valid_v has size 2 and they are compatible.
                    # If size > 2, likely there is a compatible pair unless the graph is complete.
                    # Let's just find one pair.
                    
                    # Iterate valid_v
                    found_pair = False
                    for k in range(len(valid_v)):
                        if found_pair:
                            break
                        v1 = valid_v[k]
                        # Check compatibility of v1 with rest of valid_v
                        # compatible_rest = valid_v[valid_v != v1]
                        # Check Conflict_bool[v1, compatible_rest] == 0
                        # Since valid_v are all compatible with S, we just need v1, v2 compatible with each other.
                        
                        # Vectorized check for v1 against valid_v[k+1:]
                        rest_v = valid_v[k+1:]
                        if len(rest_v) == 0:
                            continue
                        comp_mask = Conflict_bool[v1, rest_v] == 0
                        if comp_mask.any():
                            v2 = rest_v[np.where(comp_mask)[0][0]]
                            # Swap u with v1, v2
                            selected_set.add(v1)
                            selected_set.add(v2)
                            improved = True
                            found_pair = True
                            break
                            
                    if found_pair:
                        continue # Restart local search loop with new set
                
                # If no pair found, put u back
                selected_set.add(u)
                
        return list(selected_set)

    # Main Loop: Multiple Restarts
    num_restarts = 1000
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Greedy
        res = run_greedy(s)
        if not res:
            continue
        
        # 2. Local Search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
            # Optional: If we found a very good set, we can spend more time on it
            # But simple restarts are usually robust.
            
    if not best_set:
        # Fallback
        best_set = [0]
        
    return candidates[best_set].astype(np.int64)

# EVOLVE-BLOCK-END
