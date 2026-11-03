import numpy as np
import itertools
import time
from scipy.optimize import linprog

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0 (squared norm 2) - wait, this gives norm sqrt(2), not 2
    # Actually, the task says "minimum pairwise distance is at least the maximum vector norm"
    # If we use vectors of different norms, the max norm is 2 (from type 1), so min distance must be >= 2
    # A vector with two 1s has norm sqrt(2) ~ 1.414, which is less than 2, so it's valid as long as
    # its distance to all other vectors is >= 2.
    
    # Let me reconsider. The constraint is: for all pairs i,j, ||v_i - v_j|| >= max_k ||v_k||
    # If max norm is 2, then all pairwise distances must be >= 2.
    # ||v_i - v_j||^2 >= 4 for all i != j.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j >= 4
    
    # For two type-1 vectors (norm 2 each): 4 + 4 - 2*v_i·v_j >= 4 => v_i·v_j <= 2
    # For a type-1 (norm 2) and type-2 (norm sqrt(2)): 4 + 2 - 2*v_i·v_j >= 4 => v_i·v_j <= 1
    # For two type-2 vectors (norm sqrt(2) each): 2 + 2 - 2*v_i·v_j >= 4 => v_i·v_j <= 0
    
    # Let's generate all vectors with squared norm 4 or 2
    # Type 1: one 2, rest 0 (squared norm 4)
    # Type 2: two 1s, rest 0 (squared norm 2)
    # Type 3: four 1s, rest 0 (squared norm 4)
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    candidates = np.unique(candidates, axis=0)
    N = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: ||v_i - v_j||^2 < 4, i.e., ||v_i||^2 + ||v_j||^2 - 2*M[i,j] < 4
    # i.e., M[i,j] > (||v_i||^2 + ||v_j||^2 - 4) / 2
    norms_sq_outer = (norms_sq[:, None] + norms_sq[None, :]) / 2.0
    threshold = norms_sq_outer - 2.0  # M[i,j] > threshold means conflict
    Conflict = (M > threshold + 1e-9).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    NonConflict = (Conflict == 0).astype(np.uint8)
    np.fill_diagonal(NonConflict, 0)
    
    best_set = []
    best_size = 0
    
    def run_greedy_min_deg(seed):
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
                
            # MIN-degree heuristic: pick the vertex with the lowest degree
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

    def local_search_1_for_2(selected_indices, max_iters=50):
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
            rng_ls = np.random.default_rng(iters + 1000 + seed_offset)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                S_mask[i] = False
                if np.any(S_mask):
                    conflicts_with_rest = Conflict[:, S_mask].any(axis=1)
                else:
                    conflicts_with_rest = np.zeros(N, dtype=bool)
                
                compatible = ~conflicts_with_rest & ~S_mask
                
                C_indices = np.where(compatible)[0]
                if len(C_indices) < 2:
                    S_mask[i] = True
                    continue
                
                found_pair = False
                for _ in range(30):
                    if time.time() > deadline:
                        return list(selected_set)
                    j_idx = rng_ls.choice(len(C_indices))
                    j = C_indices[j_idx]
                    valid_ks = np.where(NonConflict[j, C_indices] == 1)[0]
                    if len(valid_ks) > 0:
                        k_idx = rng_ls.choice(valid_ks)
                        k = C_indices[k_idx]
                        
                        selected_set.remove(i)
                        selected_set.add(j)
                        selected_set.add(k)
                        
                        S_mask[i] = False
                        S_mask[j] = True
                        S_mask[k] = True
                        
                        improved = True
                        found_pair = True
                        break
                    
                if not found_pair:
                    S_mask[i] = True
                    continue
                    
                if found_pair:
                    break
                
        return list(selected_set)
    
    global seed_offset
    seed_offset = 0
    
    # Run greedy with multiple seeds
    for s in range(800):
        if time.time() > deadline:
            break
        
        seed_offset = s
        res = run_greedy_min_deg(s)
        if not res:
            continue
            
        res_improved = local_search_1_for_2(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(min(22, N)))
        
    return candidates[best_set].astype(np.int64)
