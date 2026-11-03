import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    D = 11
    
    # Generate ALL nonzero integer vectors with squared norm <= 4
    candidates = []
    
    # Type 1: One 1, rest 0 (squared norm = 1)
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0 (squared norm = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: One 2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 4: Four 1s, rest 0 (squared norm = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Build conflict matrix: conflict if distance_sq < 4
    # distance_sq = norms_sq[i] + norms_sq[j] - 2*M[i,j]
    norms_sq_matrix = norms_sq[:, None].astype(np.int64) + norms_sq[None, :].astype(np.int64)
    dist_sq = norms_sq_matrix - 2 * M.astype(np.int64)
    Conflict = (dist_sq < 4).astype(np.uint8)
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

    def local_search_add(selected_list):
        """Try to add compatible vectors to the selected set."""
        if time.time() > deadline:
            return selected_list
            
        selected_set = set(selected_list)
        improved = True
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            if not selected_set:
                # Pick any vector
                idx = 0
                selected_set.add(idx)
                improved = True
                continue
                
            sel_arr = np.array(list(selected_set), dtype=np.int64)
            conflicts_with_selected = Conflict[sel_arr, :].max(axis=0).astype(bool)
            conflicts_with_selected[list(selected_set)] = False
            
            candidate_indices = np.where(~conflicts_with_selected)[0]
            
            if len(candidate_indices) == 0:
                break
            
            # Try adding in random order
            rng = np.random.default_rng()
            rng.shuffle(candidate_indices)
            
            for cidx in candidate_indices:
                selected_set.add(cidx)
                improved = True
                break
                
        return list(selected_set)

    def local_search_swap(selected_list):
        """Try 1-for-2 swaps: remove one, add two compatible ones."""
        if time.time() > deadline:
            return selected_list
            
        selected_set = set(selected_list)
        if len(selected_set) < 2:
            return list(selected_set)
            
        improved = True
        max_iters = 30
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            sel_list = list(selected_set)
            rng = np.random.default_rng()
            rng.shuffle(sel_list)
            
            for i_idx in range(len(sel_list)):
                if time.time() > deadline:
                    break
                    
                i = sel_list[i_idx]
                
                # Find neighbors of i
                neighbors = np.where(Conflict[i])[0]
                valid_js = [j for j in neighbors if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                valid_js_arr = np.array(valid_js, dtype=np.int64)
                
                # Create mask for selected_set \ {i}
                rest_mask = np.zeros(N, dtype=bool)
                rest_mask[list(selected_set)] = True
                rest_mask[i] = False
                
                # Check which valid_js are compatible with all of rest
                conflict_with_rest = Conflict[valid_js_arr, :].sum(axis=1)
                compatible_js = valid_js_arr[conflict_with_rest == 0]
                
                if len(compatible_js) < 2:
                    continue
                
                # Find a compatible pair
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

    # Phase 1: Multiple greedy restarts with local search
    num_restarts = 150
    for s in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        res = run_greedy(s * 7 + 13)
        if not res:
            continue
        
        # Apply local search add
        res = local_search_add(res)
        # Apply local search swap
        res = local_search_swap(res)
        
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
    
    # Phase 2: More focused local search on best solution
    if best_size > 0 and time.time() < deadline - 10:
        # Multiple rounds of local search
        for round_num in range(5):
            if time.time() > deadline - 5:
                break
            best_set = local_search_add(best_set)
            best_set = local_search_swap(best_set)
            if len(best_set) > best_size:
                best_size = len(best_set)
    
    # Phase 3: Try to improve with different random orders
    if time.time() < deadline - 5:
        for s in range(50):
            if time.time() > deadline - 3:
                break
            
            # Start from a subset of the best solution and regrow
            if best_size > 10:
                rng = np.random.default_rng(s * 100 + 5)
                subset_size = max(5, best_size // 2)
                subset = list(rng.choice(best_set, size=subset_size, replace=False))
                res = local_search_add(subset)
                res = local_search_swap(res)
            else:
                res = run_greedy(s * 100 + 5)
                res = local_search_add(res)
                res = local_search_swap(res)
                
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
    
    if not best_set:
        best_set = [0]
        
    return candidates[best_set].astype(np.int64)
