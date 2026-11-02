import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
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
    # Distance squared = ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v = 4 + 4 - 2u.v = 8 - 2u.v
    # We need distance >= max_norm = 2.
    # distance^2 >= 4 => 8 - 2u.v >= 4 => 2u.v <= 4 => u.v <= 2.
    # So conflict if u.v > 2.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    # Helper: Run greedy
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

    # Helper: Vectorized Local Search
    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        S = set(selected_indices)
        in_S = np.zeros(N, dtype=bool)
        in_S[list(S)] = True
        
        # conf_counts[j] = number of neighbors of j in S
        conf_counts = Conflict[in_S].sum(axis=0).astype(np.int64)
        
        improved = True
        max_iterations = 5000
        iter_count = 0
        rng = np.random.default_rng(42)
        
        while improved and time.time() < deadline and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Phase 1: Try 1-for-2 swaps (remove i, add j1, j2)
            # This increases size by 1.
            s_list = list(S)
            rng.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                
                # Valid j: not in S, and conf_counts[j] == Conflict[i, j]
                # This means j only conflicts with i in S.
                # If we remove i, j has 0 conflicts with S\{i}.
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                
                if len(valid_js) >= 2:
                    # We need a pair (j1, j2) in valid_js that are compatible (Conflict[j1, j2] == 0)
                    # To speed up, if valid_js is large, sample
                    if len(valid_js) > 100:
                        sample_idx = rng.choice(len(valid_js), size=100, replace=False)
                        valid_js_check = valid_js[sample_idx]
                    else:
                        valid_js_check = valid_js
                    
                    found_pair = False
                    # Check pairs in valid_js_check
                    # Conflict[j1, j2] == 0
                    sub_conflict = Conflict[np.ix_(valid_js_check, valid_js_check)]
                    # Find any pair (p, q) with p < q and sub_conflict[p, q] == 0
                    # We can just iterate or use matrix operations
                    # Since size is small (<=100), iteration is fine
                    L = len(valid_js_check)
                    for p in range(L):
                        if found_pair:
                            break
                        # q > p
                        compat_qs = np.where(sub_conflict[p, p+1:] == 0)[0]
                        if len(compat_qs) > 0:
                            q = compat_qs[0] + p + 1
                            j1 = valid_js_check[p]
                            j2 = valid_js_check[q]
                            
                            # Perform swap
                            S.remove(i)
                            S.add(j1)
                            S.add(j2)
                            
                            in_S[i] = False
                            in_S[j1] = True
                            in_S[j2] = True
                            
                            conf_counts -= Conflict[i]
                            conf_counts += Conflict[j1]
                            conf_counts += Conflict[j2]
                            
                            improved = True
                            found_pair = True
                            break
                            
            # Phase 2: Try 1-for-1 swaps (remove i, add j)
            # This keeps size same, helps escape local minima.
            if not improved:
                s_list2 = list(S)
                rng.shuffle(s_list2)
                for i in s_list2:
                    if time.time() > deadline:
                        return list(S)
                    valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                    if len(valid_js) > 0:
                        # Pick random j
                        j = valid_js[rng.integers(len(valid_js))]
                        S.remove(i)
                        S.add(j)
                        in_S[i] = False
                        in_S[j] = True
                        conf_counts -= Conflict[i]
                        conf_counts += Conflict[j]
                        improved = True
                        break
                        
        return list(S)

    # Main loop: Try multiple seeds
    num_seeds = 100
    for s in range(num_seeds):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved

    if not best_set:
        # Fallback: just take first 22 (Type 1 vectors)
        # Type 1 vectors are 0..21
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
