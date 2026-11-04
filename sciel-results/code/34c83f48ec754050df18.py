import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    D = 11

    # =============================================================================
    # Strategy: Work with the E8 x D3 lattice structure
    # E8 has 240 minimal vectors of norm 2 (or norm 4 if we double them)
    # D3 has 12 minimal vectors of norm sqrt(2) (or norm 2 if we double)
    # 
    # Key insight: The kissing number in R^11 for D3 x D3 x Z is known to be larger
    # than just the 240 E8 vectors.
    #
    # Let's generate vectors from the D3 lattice structure and optimize.
    # =============================================================================

    # Generate all candidate vectors
    # We'll work with vectors that have small integer entries
    
    candidates = []
    candidate_set = set()

    # Add all vectors with L2 norm between 1 and 4 (inclusive) with integer entries
    # in the range [-3, 3]

    # Type 1: Exactly one non-zero entry (values ±1, ±2, ±3)
    for i in range(D):
        for v in [-3, -2, -1, 1, 2, 3]:
            vec = [0] * D
            vec[i] = v
            t = tuple(vec)
            if t not in candidate_set:
                candidate_set.add(t)
                candidates.append(vec)

    # Type 2: Exactly two non-zero entries
    for i in range(D):
        for j in range(i + 1, D):
            for vi in [-3, -2, -1, 1, 2, 3]:
                for vj in [-3, -2, -1, 1, 2, 3]:
                    # Limit norm
                    if vi * vi + vj * vj > 16:  # norm^2 > 4
                        continue
                    vec = [0] * D
                    vec[i] = vi
                    vec[j] = vj
                    t = tuple(vec)
                    if t not in candidate_set:
                        candidate_set.add(t)
                        candidates.append(vec)

    # Type 3: Exactly three non-zero entries
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for vi in [-2, -1, 1, 2]:
                    for vj in [-2, -1, 1, 2]:
                        for vk in [-2, -1, 1, 2]:
                            if vi * vi + vj * vj + vk * vk > 9:
                                continue
                            vec = [0] * D
                            vec[i] = vi
                            vec[j] = vj
                            vec[k] = vk
                            t = tuple(vec)
                            if t not in candidate_set:
                                candidate_set.add(t)
                                candidates.append(vec)

    # Type 4: Exactly four non-zero entries
    for support in itertools.combinations(range(D), 4):
        i, j, k, l = support
        for vi in [-2, -1, 1, 2]:
            for vj in [-2, -1, 1, 2]:
                for vk in [-2, -1, 1, 2]:
                    for vl in [-2, -1, 1, 2]:
                        if vi * vi + vj * vj + vk * vk + vl * vl > 16:
                            continue
                        vec = [0] * D
                        vec[i] = vi
                        vec[j] = vj
                        vec[k] = vk
                        vec[l] = vl
                        t = tuple(vec)
                        if t not in candidate_set:
                            candidate_set.add(t)
                            candidates.append(vec)

    # Type 5: Five non-zero entries (mostly ±1)
    for support in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            vec = [0] * D
            for idx, s in zip(support, signs):
                vec[idx] = s
            t = tuple(vec)
            if t not in candidate_set:
                candidate_set.add(t)
                candidates.append(vec)

    # Convert to numpy
    cand_arr = np.array(candidates, dtype=np.int64)
    N_cand = len(cand_arr)
    
    # Compute squared norms
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # Filter: only keep vectors with norm_sq >= 1
    valid_mask = norms_sq >= 1
    cand_arr = cand_arr[valid_mask]
    norms_sq = norms_sq[valid_mask]
    N_cand = len(cand_arr)
    
    # Map from vector tuple to index
    vec_to_idx = {}
    for i in range(N_cand):
        vec_to_idx[tuple(cand_arr[i])] = i

    # =============================================================================
    # Greedy construction with multiple restarts and different orderings
    # =============================================================================
    
    best_set = set()
    best_size = 0
    
    rng = np.random.RandomState(12345)
    
    # Phase 1: Greedy with different ordering strategies
    orderings = []
    
    # Ordering 1: By norm (ascending)
    orderings.append(np.argsort(norms_sq))
    # Ordering 2: By norm (descending)  
    orderings.append(-np.argsort(-norms_sq))
    # Ordering 3: Random
    for _ in range(10):
        order = np.arange(N_cand)
        rng.shuffle(order)
        orderings.append(order)
    
    def check_compatibility(v_idx, selected_indices, cand_arr, norms_sq):
        """Check if v_idx can be added to the set given by selected_indices."""
        v = cand_arr[v_idx]
        nv = norms_sq[v_idx]
        new_max = 0
        for u_idx in selected_indices:
            new_max = max(new_max, norms_sq[u_idx])
        new_max = max(new_max, nv)
        
        v_sq = nv
        for u_idx in selected_indices:
            u = cand_arr[u_idx]
            dot = np.dot(v, u)
            dist_sq = v_sq + norms_sq[u_idx] - 2 * dot
            if dist_sq < new_max:
                return False
        return True
    
    # Fast compatibility check using numpy
    def fast_check(v, v_norm_sq, selected_arr, selected_norms_sq):
        """Fast check: can v be added to selected_arr?"""
        if len(selected_arr) == 0:
            return True
        new_max = max(v_norm_sq, np.max(selected_norms_sq))
        dots = selected_arr @ v
        dist_sqs = v_norm_sq + selected_norms_sq - 2 * dots
        return np.all(dist_sqs >= new_max)
    
    for order in orderings:
        if time.time() > deadline - 40:
            break
        
        selected = []
        selected_arr = []
        selected_norms = []
        
        for v_idx in order:
            if time.time() > deadline - 40:
                break
            
            v = cand_arr[v_idx]
            nv = norms_sq[v_idx]
            
            if len(selected_arr) > 0:
                new_max = max(nv, np.max(selected_norms))
                dots = np.array(selected_arr) @ v
                dist_sqs = nv + np.array(selected_norms) - 2 * dots
                if not np.all(dist_sqs >= new_max):
                    continue
            
            selected.append(v_idx)
            selected_arr.append(v)
            selected_norms.append(nv)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Convert best_set to array for local search
    if best_size > 0:
        best_list = list(best_set)
        best_arr = cand_arr[best_list]
        best_norms = norms_sq[best_list]
        
        # Phase 2: Local search - try to add more vectors
        non_selected = [i for i in range(N_cand) if i not in best_set]
        rng.shuffle(non_selected)
        
        improved = True
        max_iterations = 200
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() > deadline - 10:
                break
            
            for v_idx in non_selected:
                if time.time() > deadline - 10:
                    break
                
                v = cand_arr[v_idx]
                nv = norms_sq[v_idx]
                
                # Find conflicts
                conflicts = []
                if len(best_arr) > 0:
                    new_max = max(nv, np.max(best_norms))
                    dots = best_arr @ v
                    dist_sqs = nv + best_norms - 2 * dots
                    conflict_mask = dist_sqs < new_max
                    for i in range(len(best_arr)):
                        if conflict_mask[i]:
                            conflicts.append(best_list[i])
                
                if len(conflicts) == 0:
                    # Add it
                    best_list.append(v_idx)
                    best_arr = np.vstack([best_arr, cand_arr[v_idx]]) if len(best_arr) > 0 else cand_arr[v_idx].reshape(1, -1)
                    best_norms = np.append(best_norms, nv)
                    best_set.add(v_idx)
                    non_selected.remove(v_idx)
                    improved = True
                    break
                else:
                    # Try removing 1 conflict
                    if len(conflicts) == 1:
                        rem = conflicts[0]
                        rem_idx = best_list.index(rem)
                        
                        # Check if v fits with rest
                        remaining_mask = np.ones(len(best_arr), dtype=bool)
                        remaining_mask[rem_idx] = False
                        rem_arr = best_arr[remaining_mask]
                        rem_norms = best_norms[remaining_mask]
                        
                        if len(rem_arr) > 0:
                            new_max = max(nv, np.max(rem_norms))
                            dots = rem_arr @ v
                            dist_sqs = nv + rem_norms - 2 * dots
                            if np.all(dist_sqs >= new_max):
                                # Perform the swap
                                best_list.remove(rem)
                                best_arr = rem_arr
                                best_norms = rem_norms
                                best_set.discard(rem)
                                best_list.append(v_idx)
                                best_arr = np.vstack([best_arr, cand_arr[v_idx]]) if len(best_arr) > 0 else cand_arr[v_idx].reshape(1, -1)
                                best_norms = np.append(best_norms, nv)
                                best_set.add(v_idx)
                                non_selected.remove(v_idx)
                                improved = True
                                break
                    elif len(conflicts) <= 3:
                        # Try removing 2 conflicts
                        for i in range(len(conflicts)):
                            for j in range(i + 1, len(conflicts)):
                                rem1, rem2 = conflicts[i], conflicts[j]
                                idx1 = best_list.index(rem1)
                                idx2 = best_list.index(rem2)
                                
                                remaining_mask = np.ones(len(best_arr), dtype=bool)
                                remaining_mask[[idx1, idx2]] = False
                                rem_arr = best_arr[remaining_mask]
                                rem_norms = best_norms[remaining_mask]
                                
                                if len(rem_arr) > 0:
                                    new_max = max(nv, np.max(rem_norms))
                                    dots = rem_arr @ v
                                    dist_sqs = nv + rem_norms - 2 * dots
                                    if np.all(dist_sqs >= new_max):
                                        for rem in [rem1, rem2]:
                                            best_list.remove(rem)
                                            best_set.discard(rem)
                                        best_arr = rem_arr
                                        best_norms = rem_norms
                                        best_list.append(v_idx)
                                        best_arr = np.vstack([best_arr, cand_arr[v_idx]])
                                        best_norms = np.append(best_norms, nv)
                                        best_set.add(v_idx)
                                        non_selected.remove(v_idx)
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
    
    # Final result
    if best_size == 0:
        # Fallback
        result = []
        for i in range(D):
            for s in [2, -2]:
                v = [0] * D
                v[i] = s
                result.append(v)
        return np.array(result, dtype=np.int64)
    
    return cand_arr[sorted(best_set)]
