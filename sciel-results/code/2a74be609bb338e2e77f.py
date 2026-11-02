"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W2 vectors (weight 2, entries in {-1,0,1}, norm^2=2)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    N2 = len(vectors_v2)
    N1 = len(vectors_v1)
    
    if N4 == 0:
        return np.vstack([vectors_v2, vectors_v1])
        
    # Compute conflict matrix for W4: conflict if dot > 2
    dots44 = vectors_v4 @ vectors_v4.T
    conflicts44 = dots44 > 2
    np.fill_diagonal(conflicts44, False)
    
    # Compatibility matrix for W4
    compat44 = ~conflicts44
    np.fill_diagonal(compat44, True)
    
    # Precompute neighbor lists for W4
    neighbors44 = [np.where(conflicts44[i])[0] for i in range(N4)]
    neighbor_counts44 = np.array([len(n) for n in neighbors44], dtype=np.int32)
    
    # Expansion scores
    expansion_scores44 = np.zeros(N4, dtype=np.int32)
    for i in range(N4):
        nbrs = neighbors44[i]
        if len(nbrs) > 1:
            sub_conf = conflicts44[np.ix_(nbrs, nbrs)]
            expansion_scores44[i] = sub_conf.sum() // 2
    
    rng = np.random.default_rng(12345)
    
    best_size4 = 0
    best_in_set4 = np.zeros(N4, dtype=bool)
    
    def greedy_weighted_init(seed=None):
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
            
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts44.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            degs = degrees[alive_indices]
            mins = np.min(degs)
            cands = alive_indices[degs == mins]
            
            if len(cands) > 1:
                exps = expansion_scores44[cands]
                min_exp = np.min(exps)
                final_cands = cands[exps == min_exp]
                chosen = final_cands[local_rng.integers(len(final_cands))]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors44[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts44[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search4(in_set_init, max_time=8.0, T_start=2.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = T_start
        cooling = 0.9997
        min_T = 0.02
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors44[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                
                if len(min_cands) > 1:
                    degs = neighbor_counts44[min_cands]
                    min_deg = np.min(degs)
                    v = min_cands[degs == min_deg][rng.integers(np.sum(degs == min_deg))]
                else:
                    v = min_cands[0]
                
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts44[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors44[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors44[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors44[v]] -= 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                n_sample = min(15, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_cands = not_in_set[sample_idx]
                
                temp_counts = count_conflicts.copy()
                temp_counts[neighbors44[v_remove]] -= 1
                temp_counts[v_remove] = 0
                
                cands_conf = temp_counts[sample_cands]
                min_conf = np.min(cands_conf)
                good_cands = sample_cands[cands_conf == min_conf]
                
                if len(good_cands) > 0:
                    v_add = good_cands[rng.integers(len(good_cands))]
                    delta = 1 - int(min_conf)
                    
                    if delta >= 0 or rng.random() < np.exp(delta / T):
                        to_remove = np.where(conflicts44[v_add] & in_set)[0]
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors44[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors44[v_add]] += 1
                else:
                    stuck_counter += 1
                    
            else:
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 == v2:
                        v2 = in_set_indices[rng.integers(len(in_set_indices))]
                        if v1 == v2:
                            stuck_counter += 1
                        else:
                            n_sample = min(10, len(not_in_set))
                            sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                            sample_cands = not_in_set[sample_idx]
                            
                            found = []
                            for v_add in sample_cands:
                                conflicts_with = np.where(conflicts44[v_add] & in_set)[0]
                                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                                if len(other_conflicts) == 0:
                                    found.append(v_add)
                                    if len(found) == 2:
                                        break
                            
                            if len(found) == 2:
                                to_remove = np.array([v1, v2], dtype=np.int64)
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors44[u]] -= 1
                                for v_ad in found:
                                    in_set[v_ad] = True
                                    count_conflicts[neighbors44[v_ad]] += 1
                            else:
                                stuck_counter += 1
                    else:
                        n_sample = min(10, len(not_in_set))
                        sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                        sample_cands = not_in_set[sample_idx]
                        
                        found = []
                        for v_add in sample_cands:
                            conflicts_with = np.where(conflicts44[v_add] & in_set)[0]
                            other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                            if len(other_conflicts) == 0:
                                found.append(v_add)
                                if len(found) == 2:
                                    break
                        
                        if len(found) == 2:
                            to_remove = np.array([v1, v2], dtype=np.int64)
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors44[u]] -= 1
                            for v_ad in found:
                                in_set[v_ad] = True
                                count_conflicts[neighbors44[v_ad]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def greedy_polish4(in_set_init):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors44[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            addable = not_in[count_conflicts[not_in] == 0]
            
            if len(addable) > 0:
                v_idx = addable[0]
                in_set[v_idx] = True
                current_size += 1
                count_conflicts[neighbors44[v_idx]] += 1
                changed = True
        
        return in_set, current_size
    
    # Main search loop for W4
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        seed = n_restarts * 7 + 13
        in_set = greedy_weighted_init(seed=seed)
        current_size = int(in_set.sum())
        
        if current_size > best_size4:
            best_size4 = current_size
            best_in_set4 = in_set.copy()
            
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        sa_time = min(remaining_time * 0.5, 8.0)
        sa_result, sa_size = sa_local_search4(in_set, max_time=sa_time)
        
        if sa_size > best_size4:
            best_size4 = sa_size
            best_in_set4 = sa_result.copy()
            
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2:
            polish_set, polish_size = greedy_polish4(sa_result)
            if polish_size > best_size4:
                best_size4 = polish_size
                best_in_set4 = polish_set.copy()
    
    # Get selected W4 vectors
    final_v4_indices = np.where(best_in_set4)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Filter W2 vectors compatible with selected W4
    # Condition: dot <= 1
    if len(result_v4) > 0:
        dots24 = vectors_v2 @ result_v4.T
        # Conflict if any dot > 1
        conflicts24 = (dots24 > 1).any(axis=1)
    else:
        conflicts24 = np.zeros(N2, dtype=bool)
        
    valid_w2_indices = np.where(~conflicts24)[0]
    vectors_v2_valid = vectors_v2[valid_w2_indices]
    N2_valid = len(vectors_v2_valid)
    
    # Solve MIS for W2 valid
    # W2 graph: conflict if dot > 0
    if N2_valid > 0:
        dots22 = vectors_v2_valid @ vectors_v2_valid.T
        conflicts22 = dots22 > 0
        np.fill_diagonal(conflicts22, False)
        
        # Simple greedy for W2 (small graph)
        in_set2 = np.zeros(N2_valid, dtype=bool)
        alive2 = np.ones(N2_valid, dtype=bool)
        # Greedy by degree
        degs2 = np.sum(conflicts22, axis=1)
        order2 = np.argsort(degs2)
        
        for idx in order2:
            if not alive2[idx]:
                continue
            in_set2[idx] = True
            # Remove neighbors
            nbrs = np.where(conflicts22[idx])[0]
            alive2[nbrs] = False
            alive2[idx] = False
            
        result_v2 = vectors_v2_valid[in_set2]
    else:
        result_v2 = np.empty((0, D), dtype=np.int64)
        
    # Filter W1 vectors compatible with selected W4 and W2
    # W1 vs W4: always compatible (dot <= 2)
    # W1 vs W2: conflict if dot > 1
    if len(result_v2) > 0:
        dots12 = vectors_v1 @ result_v2.T
        conflicts12 = (dots12 > 1).any(axis=1)
    else:
        conflicts12 = np.zeros(N1, dtype=bool)
        
    valid_w1_indices = np.where(~conflicts12)[0]
    result_v1 = vectors_v1[valid_w1_indices]
    
    # Combine
    final_vectors = np.vstack([result_v4, result_v2, result_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
