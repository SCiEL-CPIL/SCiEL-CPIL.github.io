import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (evaluator kills at 150s)
    
    # Generate candidates with squared norm 4 and 8
    candidates = []
    
    # Type 1: One 2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (norm^2 = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type 3: Two 2s, rest 0 (norm^2 = 8)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = 2 * s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute norms and dot product matrix
    norms_sq = np.sum(candidates**2, axis=1)  # shape (N,)
    M = candidates @ candidates.T  # shape (N, N)
    
    # Conflict matrix: conflict if ||v_i - v_j||^2 < max_norm_sq
    # We'll use max_norm_sq = 8 (the maximum norm^2 in our candidate set)
    # ||v_i - v_j||^2 = norms_sq[i] + norms_sq[j] - 2*M[i,j]
    # Conflict if norms_sq[i] + norms_sq[j] - 2*M[i,j] < 8
    # i.e., M[i,j] > (norms_sq[i] + norms_sq[j] - 8) / 2
    max_norm_sq = 8
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    Conflict = (M > threshold).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Precompute neighbor lists for faster local search
    neighbor_lists = [np.where(Conflict[i])[0] for i in range(N)]
    
    # Greedy construction with min-degree heuristic
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
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Efficient local search using bitmasks
    def local_search(selected_indices, max_iterations=50):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iterations:
            improved = False
            iter_count += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Get neighbors of i that are not in selected_set
                neigh_i = neighbor_lists[i]
                valid_js = [j for j in neigh_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each j in valid_js, check if j is compatible with all of S\{i}
                # Use vectorized check: Conflict[j, selected_set] should only have i
                s_list = list(selected_set)
                s_arr = np.array(s_list)
                
                # Conflict with all selected: shape (len(valid_js), len(s_list))
                conflicts_with_s = Conflict[np.array(valid_js)[:, np.newaxis], s_arr[np.newaxis, :]]
                
                # Find j's that conflict only with i (or not at all, but they're neighbors so they conflict with i)
                # Count conflicts with each j
                conflict_counts = conflicts_with_s.sum(axis=1)
                
                # j must have exactly 1 conflict (with i) and that conflict must be with i
                # Find index of i in s_list
                i_pos = s_list.index(i)
                
                # j's with conflict_count == 1 and the conflict is at position i_pos
                single_conflict_mask = (conflict_counts == 1)
                if not np.any(single_conflict_mask):
                    continue
                    
                # Among those, check that the conflict is specifically with i
                candidate_js = valid_js[single_conflict_mask]
                compatible_js = []
                for j in candidate_js:
                    if conflicts_with_s[np.array(valid_js).tolist().index(j), i_pos]:
                        compatible_js.append(j)
                
                if len(compatible_js) < 2:
                    continue
                
                # Find a compatible pair among compatible_js
                compat_arr = np.array(compatible_js)
                sub_conflict = Conflict[np.ix_(compat_arr, compat_arr)]
                np.fill_diagonal(sub_conflict, 0)
                
                # Find pairs with no conflict
                found = False
                for a in range(len(compat_arr)):
                    if found:
                        break
                    for b in range(a + 1, len(compat_arr)):
                        if sub_conflict[a, b] == 0:
                            selected_set.remove(i)
                            selected_set.add(compat_arr[a])
                            selected_set.add(compat_arr[b])
                            improved = True
                            found = True
                            break
                            
        return list(selected_set)

    # Try multiple greedy runs with local search
    best_size = 0
    best_set = []
    
    # First pass: quick greedy to get a baseline
    for seed in range(20):
        if time.time() > deadline:
            break
        res = run_greedy(seed)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
    
    # Second pass: more greedy runs with local search on promising ones
    for seed in range(20, 100):
        if time.time() > deadline:
            break
        res = run_greedy(seed)
        
        # Only do local search if this is competitive
        if len(res) >= best_size - 2:
            improved_set = local_search(res)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        elif len(res) > best_size:
            best_size = len(res)
            best_set = res
    
    # Final polish: try local search on the best set
    if best_size > 0 and time.time() < deadline - 5:
        polished = local_search(best_set, max_iterations=100)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished

    # Convert indices back to vectors
    if best_size > 0:
        return candidates[best_set].astype(np.int64)
    else:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:5])
    norms = np.sum(res**2, axis=1)
    max_norm = np.max(norms)
    dists = np.sum((res[:, np.newaxis, :] - res[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    print(f"Max Norm: {max_norm}, Min Dist: {min_dist}, Valid: {min_dist >= max_norm}")
    print(f"Count: {len(res)}")
