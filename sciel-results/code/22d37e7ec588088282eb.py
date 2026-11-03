import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # NonConflict matrix for faster pair finding
    # NonConflict[i, j] = 1 if i != j and not Conflict[i, j]
    NonConflict = (Conflict == 0).astype(np.uint8)
    np.fill_diagonal(NonConflict, 0)
    
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
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices, max_iters=50):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        S_mask = np.zeros(N, dtype=bool)
        for i in selected_set:
            S_mask[i] = True
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters + 1000)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Calculate compatible nodes for removing i
                # S_without_i
                S_mask[i] = False
                # conflicts_with_S_minus_i: True if j conflicts with any node in S \ {i}
                # Conflict[:, S_mask] has shape (N, |S|)
                # We want any() along axis 1
                if np.any(S_mask):
                    conflicts_with_rest = Conflict[:, S_mask].any(axis=1)
                else:
                    conflicts_with_rest = np.zeros(N, dtype=bool)
                
                # Compatible: not in S, and not conflicting with S \ {i}
                compatible = ~conflicts_with_rest & ~S_mask
                
                # Restore S_mask[i] for next iteration logic or just keep it modified?
                # We are checking swap, so we assume i is removed.
                # If we find a swap, we will update S_mask.
                
                C_indices = np.where(compatible)[0]
                if len(C_indices) < 2:
                    S_mask[i] = True # Restore if no swap
                    continue
                
                # Find a pair j, k in C_indices such that NonConflict[j, k] == 1
                # Strategy: Pick random j from C_indices, check if any k works.
                found_pair = False
                # Try a few random picks to speed up
                for _ in range(10):
                    if time.time() > deadline:
                        return list(selected_set)
                    j_idx = rng_ls.choice(len(C_indices))
                    j = C_indices[j_idx]
                    # Check NonConflict[j, C_indices]
                    valid_ks = np.where(NonConflict[j, C_indices] == 1)[0]
                    # Exclude j itself (though NonConflict diagonal is 0, so j won't be there)
                    if len(valid_ks) > 0:
                        k_idx = rng_ls.choice(valid_ks)
                        k = C_indices[k_idx]
                        
                        # Perform Swap
                        selected_set.remove(i)
                        selected_set.add(j)
                        selected_set.add(k)
                        
                        # Update S_mask
                        S_mask[i] = False
                        S_mask[j] = True
                        S_mask[k] = True
                        
                        improved = True
                        found_pair = True
                        break
                    # If no k found for this j, try another j
                    
                if not found_pair:
                    # Restore S_mask[i] if we didn't swap
                    S_mask[i] = True
                    continue
                    
                if found_pair:
                    break # Restart local search loop with new set
                
        return list(selected_set)

    # Run greedy with multiple seeds
    for s in range(300):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
