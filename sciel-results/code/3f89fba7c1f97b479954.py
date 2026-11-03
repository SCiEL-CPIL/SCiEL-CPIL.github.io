"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Generate all nonzero integer vectors with squared norm <= 5
    # This includes:
    # Type A: one 2 (norm^2=4)
    # Type B: two 1s (norm^2=2)
    # Type C: four 1s (norm^2=4)
    # Type D: five 1s (norm^2=5)
    # Type E: one 2 + one 1 (norm^2=5)
    # Type F: one 1 (norm^2=1)
    
    candidates = []
    
    # Type F: single 1 (norm^2 = 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Type B: two 1s (norm^2 = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type A: one 2 (norm^2 = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    
    # Type C: four 1s (norm^2 = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type D: five 1s (norm^2 = 5)
    for idxs in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type E: one 2 + one 1 (norm^2 = 5)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for s2 in [1, -1]:
                for s1 in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * s2
                    v[j] = s1
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Max squared norm in our set
    max_norm_sq = int(np.max(norms_sq))
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Build conflict matrix:
    # For pair (i,j), we need ||v_i - v_j||^2 >= max_norm_sq
    # ||v_i - v_j||^2 = norms_sq[i] + norms_sq[j] - 2*M[i,j]
    # Conflict if norms_sq[i] + norms_sq[j] - 2*M[i,j] < max_norm_sq
    
    norms_sq_matrix = norms_sq[:, None] + norms_sq[None, :]
    dist_sq = norms_sq_matrix - 2 * M
    Conflict = (dist_sq < max_norm_sq).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
        
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set))
        
        improved = True
        max_iters = 30
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            rng_ls = np.random.default_rng()
            order = rng_ls.permutation(len(selected_arr))
            selected_arr = selected_arr[order]
            
            for i_idx in range(len(selected_arr)):
                if time.time() > deadline:
                    break
                
                i = selected_arr[i_idx]
                
                # Find neighbors of i not in selected_set
                neighbors = np.where(Conflict[i])[0]
                valid_js = [j for j in neighbors if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                valid_js_arr = np.array(valid_js)
                
                # Build mask for selected_set \ {i}
                rest_mask = np.zeros(N, dtype=bool)
                rest_mask[list(selected_set)] = True
                rest_mask[i] = False
                
                # Conflict counts of each valid_j with the rest
                conflict_with_rest = Conflict[valid_js_arr, :].sum(axis=1)
                compatible_js = valid_js_arr[conflict_with_rest == 0]
                
                if len(compatible_js) < 2:
                    continue
                
                # Find a pair in compatible_js that are mutually compatible
                found = False
                for p in range(len(compatible_js)):
                    if found:
                        break
                    j1 = compatible_js[p]
                    rest_compatible = compatible_js[p+1:]
                    if len(rest_compatible) == 0:
                        continue
                    comp_with_j1 = Conflict[j1, rest_compatible] == 0
                    if comp_with_j1.any():
                        j2 = rest_compatible[np.where(comp_with_j1)[0][0]]
                        selected_set.remove(i)
                        selected_set.add(j1)
                        selected_set.add(j2)
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        return list(selected_set)

    def local_search_3swap(selected_indices):
        """Try 1-for-3 swaps: remove one vertex, add three mutually compatible non-neighbors."""
        if time.time() > deadline:
            return selected_indices
        
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set))
        
        improved = True
        max_iters = 10
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            rng_ls = np.random.default_rng()
            order = rng_ls.permutation(len(selected_arr))
            selected_arr = selected_arr[order]
            
            for i_idx in range(len(selected_arr)):
                if time.time() > deadline:
                    break
                
                i = selected_arr[i_idx]
                
                neighbors = np.where(Conflict[i])[0]
                valid_js = [j for j in neighbors if j not in selected_set]
                
                if len(valid_js) < 3:
                    continue
                
                valid_js_arr = np.array(valid_js)
                
                rest_mask = np.zeros(N, dtype=bool)
                rest_mask[list(selected_set)] = True
                rest_mask[i] = False
                
                conflict_with_rest = Conflict[valid_js_arr, :].sum(axis=1)
                compatible_js = valid_js_arr[conflict_with_rest == 0]
                
                if len(compatible_js) < 3:
                    continue
                
                # Find 3 mutually compatible vertices
                found = False
                n_comp = len(compatible_js)
                for p in range(min(n_comp, 50)):
                    if found:
                        break
                    j1 = compatible_js[p]
                    rest1 = compatible_js[p+1:]
                    if len(rest1) < 2:
                        continue
                    comp_with_j1 = Conflict[j1, rest1] == 0
                    cands1 = rest1[comp_with_j1]
                    if len(cands1) < 2:
                        continue
                    for q in range(min(len(cands1), 30)):
                        if found:
                            break
                        j2 = cands1[q]
                        rest2 = cands1[q+1:]
                        if len(rest2) < 1:
                            continue
                        comp_with_j2 = Conflict[j2, rest2] == 0
                        cands2 = rest2[comp_with_j2]
                        if len(cands2) >= 1:
                            j3 = cands2[0]
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            selected_set.add(j3)
                            improved = True
                            found = True
                            break
                
                if found:
                    break
        
        return list(selected_set)

    # Run multiple greedy restarts with local search
    num_restarts = 150
    for s in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        res = run_greedy(s)
        if not res:
            continue
        
        # Apply local search
        res_improved = local_search(res)
        
        # Try 1-for-3 swaps
        res_improved = local_search_3swap(res_improved)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
    
    if not best_set:
        # Fallback
        best_set = list(range(min(5, N)))
        
    return candidates[best_set].astype(np.int64)

# EVOLVE-BLOCK-END
