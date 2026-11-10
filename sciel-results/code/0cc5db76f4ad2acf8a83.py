import numpy as np
import time
import random
from scipy.optimize import linprog
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0  # Safety margin before 630s kill
    
    # Start with the known optimal base for norm^2=2 (weight 2 vectors)
    # These are all vectors with two non-zero entries, each +/-1.
    # Total: C(11,2) * 4 = 55 * 4 = 220 points.
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    
    current_points = np.array(base_list, dtype=np.int64)
    current_k = current_points.shape[0]
    
    # Helper to check if a new point is valid against current set
    # Condition: ||x - y||^2 >= max(||x||^2, ||y||^2) for all y in current_set
    # Let R = max(||x||^2, ||y||^2).
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y >= R
    # If ||x||^2 >= ||y||^2, then R = ||x||^2.
    #   ||x||^2 + ||y||^2 - 2 x.y >= ||x||^2  =>  ||y||^2 >= 2 x.y
    # If ||y||^2 > ||x||^2, then R = ||y||^2.
    #   ||x||^2 + ||y||^2 - 2 x.y >= ||y||^2  =>  ||x||^2 >= 2 x.y
    
    def is_valid(x, points, norms):
        x_norm_sq = np.dot(x, x)
        # For each y in points, check the condition
        # We can vectorize this
        # dots = points @ x
        # dist_sq = x_norm_sq + norms - 2 * dots
        # required = np.maximum(x_norm_sq, norms)
        # valid = np.all(dist_sq >= required)
        
        # Optimization: 
        # Case 1: x_norm_sq >= y_norm_sq for a specific y.
        #   Need y_norm_sq >= 2 * dot(x, y)
        # Case 2: y_norm_sq > x_norm_sq for a specific y.
        #   Need x_norm_sq >= 2 * dot(x, y)
        
        dots = points @ x
        dist_sq = x_norm_sq + norms - 2 * dots
        required = np.maximum(x_norm_sq, norms)
        return np.all(dist_sq >= required)

    # Precompute norms for current points
    norms = np.sum(current_points * current_points, axis=1).astype(np.int64)
    
    # We will try to add points with higher norms.
    # Strategy: Iterate through possible weights (norm^2) 3, 4, 5...
    # For each weight, generate candidate integer vectors with that norm.
    # Since the space of integer vectors with norm^2 = w is finite but grows fast,
    # we will use a randomized search guided by the validity check.
    # However, simple random search might be too slow.
    # 
    # Alternative Structural Approach: 
    # Instead of random search, we can use the fact that the problem is symmetric.
    # But let's try a "Greedy with Backtracking" on a structured pool.
    # 
    # Let's generate a pool of candidates for weights 3, 4, 5, 6.
    # Weight 3: C(11,3)*8 = 165*8 = 1320
    # Weight 4: C(11,4)*16 + C(11,2)*C(2,2)*4 (two 2s? No, entries are -1,0,1)
    # Wait, entries are integers. The prompt implies general integer points?
    # "Integer kissing configuration". 
    # The seed program used {-1, 0, 1}. 
    # If we allow entries like 2, the norm grows faster.
    # However, the constraint is tight. 
    # Let's stick to {-1, 0, 1} for now to keep the search space tractable, 
    # but use a better selection method than pure random.
    
    # Generate candidates for weights 3, 4, 5
    candidates = []
    for w in [3, 4, 5]:
        # Generate all vectors in {-1, 0, 1}^11 with weight w
        # This is equivalent to choosing w positions, then assigning signs.
        # To avoid duplicates and speed up, we can iterate through combinations of indices.
        from itertools import combinations
        for combo in combinations(range(d), w):
            # For each combo of indices, iterate through all 2^w sign assignments
            for mask in range(1 << w):
                v = np.zeros(d, dtype=np.int64)
                for idx, pos in enumerate(combo):
                    if (mask >> idx) & 1:
                        v[pos] = 1
                    else:
                        v[pos] = -1
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int64)
    cand_norms = np.sum(candidates * candidates, axis=1).astype(np.int64)
    
    # We want to select a subset of these candidates that are valid against each other
    # AND valid against the base points.
    # First, filter out candidates that conflict with the base points.
    valid_candidates = []
    valid_cand_norms = []
    
    for i in range(candidates.shape[0]):
        if is_valid(candidates[i], current_points, norms):
            valid_candidates.append(candidates[i])
            valid_cand_norms.append(cand_norms[i])
            
    if len(valid_candidates) == 0:
        return current_points
        
    valid_candidates = np.array(valid_candidates, dtype=np.int64)
    valid_cand_norms = np.array(valid_cand_norms, dtype=np.int64)
    
    # Now we have a pool of candidates that are all valid against the base.
    # We need to find the largest subset of these candidates that are mutually valid.
    # This is again an MIS problem, but on a much smaller pool (candidates valid against base).
    # The pool size for w=3,4,5 is roughly 1320 + 330*16 + 330*32? 
    # C(11,3)=165, *8=1320.
    # C(11,4)=330, *16=5280.
    # C(11,5)=462, *32=14784.
    # Total ~21k candidates.
    # Many will be filtered out by the base check.
    
    # Let's build the conflict graph among these valid candidates.
    n_cands = valid_candidates.shape[0]
    
    # If n_cands is small enough, we can use a greedy approach with local search.
    # If it's large, we might need to be smarter.
    # Given the time limit, a good greedy + local search is safer than exact MIS.
    
    # Compute pairwise conflicts among valid candidates
    # dist^2(i, j) >= max(norm_i, norm_j)
    
    # To speed up, we can compute the conflict matrix in blocks or just if n_cands < 5000.
    # 21k is too big for full matrix in Python loops, but numpy can handle 21k x 21k bools (~440MB).
    # Let's try to compute it.
    
    dots = valid_candidates @ valid_candidates.T
    dist_sq = valid_cand_norms[:, None] + valid_cand_norms[None, :] - 2 * dots
    required = np.maximum(valid_cand_norms[:, None], valid_cand_norms[None, :])
    conflict = dist_sq < required
    
    # Greedy selection from this pool
    # Order by "compatibility" (number of non-conflicting neighbors)
    deg_compat = np.sum(~conflict, axis=1) - 1
    order = np.argsort(-deg_compat)
    
    selected_cands = []
    in_selected = np.zeros(n_cands, dtype=bool)
    
    for idx in order:
        if time.time() > deadline - 50:
            break
        if not selected_cands:
            selected_cands.append(idx)
            in_selected[idx] = True
            continue
            
        # Check if idx conflicts with any selected
        if np.any(conflict[idx, selected_cands]):
            continue
            
        selected_cands.append(idx)
        in_selected[idx] = True
        
    # Merge base and selected candidates
    final_points = np.vstack([current_points, valid_candidates[selected_cands]])
    final_k = final_points.shape[0]
    
    # Local Search on the final set?
    # We can try to swap points in the final set to increase size.
    # But since we already did a greedy on the candidate pool, and the base is fixed,
    # the main gain comes from the candidate selection.
    
    # Let's try a few random restarts of the greedy on the candidate pool if time permits.
    best_k = final_k
    best_points = final_points
    
    if time.time() < deadline - 100:
        for restart in range(5):
            if time.time() > deadline - 50:
                break
            order = np.random.permutation(n_cands)
            selected_cands = []
            in_selected = np.zeros(n_cands, dtype=bool)
            
            for idx in order:
                if time.time() > deadline - 50:
                    break
                if not selected_cands:
                    selected_cands.append(idx)
                    in_selected[idx] = True
                    continue
                if np.any(conflict[idx, selected_cands]):
                    continue
                selected_cands.append(idx)
                in_selected[idx] = True
                
            candidate_final = np.vstack([current_points, valid_candidates[selected_cands]])
            if candidate_final.shape[0] > best_k:
                best_k = candidate_final.shape[0]
                best_points = candidate_final

    return best_points.astype(np.int64)
