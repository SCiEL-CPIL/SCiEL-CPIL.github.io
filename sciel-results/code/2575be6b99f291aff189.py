import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # 1. Generate candidate vectors with weights 1 to 4
    candidates = []
    # Weight 1
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = sign
            candidates.append(v)
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # Precompute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Precompute dot products? 
    # dots = candidates @ candidates.T
    # This is N x N. For N=8162, this is ~67 million entries. 
    # 67M * 8 bytes = 536 MB. Might be tight but doable.
    # Let's try to compute on the fly for each M to save memory if needed, 
    # or just compute once if memory allows.
    # 8162^2 is about 66.6 million. 
    # Let's assume we can afford it.
    
    dots = candidates @ candidates.T
    
    best_result = np.array([[1] + [0]*10], dtype=np.int64)
    best_size = 1

    # Try each possible max norm squared M
    # Possible M values are 1, 2, 3, 4
    for M in [1, 2, 3, 4]:
        if time.time() > deadline - 10:
            break
            
        # Filter candidates with norm_sq <= M
        mask = norms_sq <= M
        n_m = np.sum(mask)
        if n_m == 0:
            continue
            
        indices_m = np.where(mask)[0]
        sub_dots = dots[np.ix_(indices_m, indices_m)]
        
        # Conflict if ||u-v||^2 < M
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        # We need sub_dots[i,j] > (norms_sq[i] + norms_sq[j] - M) / 2
        # Let's compute the threshold matrix
        # thr = (norms_sq[:, None] + norms_sq[None, :] - M) / 2.0
        # conflict = sub_dots > thr
        
        # To avoid float issues, use: 2 * sub_dots > norms_sq[i] + norms_sq[j] - M
        # 2 * sub_dots - norms_sq[i] - norms_sq[j] > -M
        # norms_sq is integer, sub_dots is integer.
        
        n_sq_sub = norms_sq[indices_m]
        # Create a matrix of norms_sq[i] + norms_sq[j]
        # This is n_m x n_m
        norm_sum = n_sq_sub[:, None] + n_sq_sub[None, :]
        
        # Conflict condition: 2 * sub_dots > norm_sum - M
        # 2 * sub_dots - norm_sum > -M
        
        # Compute LHS
        lhs = 2 * sub_dots - norm_sum
        
        conflicts = lhs > -M
        np.fill_diagonal(conflicts, False)
        
        # Now find MIS in this graph of size n_m
        # Greedy + Local Search
        
        def solve_mis(conflicts, n, time_limit):
            # Greedy initialization
            best_selected = np.zeros(n, dtype=bool)
            best_count = 0
            
            rng = np.random.default_rng(42)
            
            # Multiple restarts
            num_restarts = 10
            for r in range(num_restarts):
                if time.time() > time_limit:
                    break
                
                alive = np.ones(n, dtype=bool)
                selected = np.zeros(n, dtype=bool)
                degrees = conflicts.sum(axis=1).astype(np.int64)
                
                while np.any(alive):
                    alive_indices = np.where(alive)[0]
                    if len(alive_indices) == 0:
                        break
                    alive_degrees = degrees[alive_indices]
                    min_deg = np.min(alive_degrees)
                    cands = alive_indices[alive_degrees == min_deg]
                    
                    # Tie-breaking: random
                    if len(cands) > 1:
                        # Optional: prefer lower conflict count among alive
                        # For speed, just random
                        chosen = cands[rng.integers(len(cands))]
                    else:
                        chosen = cands[0]
                        
                    selected[chosen] = True
                    # Remove chosen and its neighbors
                    to_remove = np.where(conflicts[chosen] | (np.arange(n) == chosen))[0]
                    alive[to_remove] = False
                    remaining = np.where(alive)[0]
                    if len(remaining) > 0:
                        degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
                        
                count = selected.sum()
                if count > best_count:
                    best_count = count
                    best_selected = selected.copy()
                    
            # Local Search
            # 1-swap
            for iter_ls in range(50):
                if time.time() > time_limit:
                    break
                improved = False
                non_selected = np.where(~best_selected)[0]
                if len(non_selected) == 0:
                    break
                
                # Try to add a non-selected node if it has no neighbors in selected
                # Or swap
                for v in non_selected:
                    sel_neighbors = np.where(best_selected & conflicts[v])[0]
                    if len(sel_neighbors) == 0:
                        best_selected[v] = True
                        best_count += 1
                        improved = True
                        break
                    
                    # 1-swap: remove one neighbor, add v
                    if len(sel_neighbors) == 1:
                        # Net gain 0, but might enable further moves? 
                        # Usually we want net gain > 0.
                        pass
                        
                if not improved:
                    # Try 2-swap
                    if len(non_selected) >= 2:
                        found = False
                        for _ in range(100):
                            if time.time() > time_limit:
                                break
                            i, j = rng.choice(non_selected, size=2, replace=False)
                            if conflicts[i, j]:
                                continue
                            sel_neigh_i = np.where(best_selected & conflicts[i])[0]
                            sel_neigh_j = np.where(best_selected & conflicts[j])[0]
                            combined = np.unique(np.concatenate([sel_neigh_i, sel_neigh_j]))
                            net_gain = 2 - len(combined)
                            if net_gain > 0:
                                best_selected[i] = True
                                best_selected[j] = True
                                for sn in combined:
                                    best_selected[sn] = False
                                best_count += net_gain
                                improved = True
                                found = True
                                break
                        if found:
                            continue
                            
                if not improved:
                    break
                    
            return best_selected, best_count

        # Allocate time for this M
        # Smaller M -> smaller graph -> faster.
        # Larger M -> larger graph -> slower.
        # Let's just use the remaining time budget.
        current_time = time.time()
        time_left = deadline - 5 - current_time
        if time_left <= 0:
            continue
            
        selected, count = solve_mis(conflicts, n_m, time_left)
        
        if count > best_size:
            best_size = count
            # Map back to original indices
            best_indices = indices_m[selected]
            best_result = candidates[best_indices]

    return best_result
