<<<<<<< SEARCH
# EVOLVE-BLOCK-START
P = {'n': 12, 'm': 40}

def construct():
    return [[b for b in range(P['n']) if (mask>>b)&1] for mask in range(P['m'])]
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    masks = np.arange(num_masks, dtype=np.int32)
    
    # Build comparability matrix: comp[i,j]=1 if i,j comparable (one strict subset of other)
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    def evaluate(indices):
        if len(indices) < 2:
            return 0
        sub = comp[np.ix_(indices, indices)]
        return int(sub.sum() // 2)
    
    def hill_climb(current_indices, time_limit):
        current_indices = np.array(current_indices, dtype=np.int32)
        current_score = evaluate(current_indices)
        
        in_set = set(current_indices.tolist())
        not_in_set = np.array([x for x in range(num_masks) if x not in in_set], dtype=np.int32)
        
        improved = True
        max_iter = 500
        iter_count = 0
        while improved and time.time() < time_limit and iter_count < max_iter:
            improved = False
            iter_count += 1
            in_list = current_indices.tolist()
            
            # loss_vec[i_idx] = comp[i, in_set].sum()
            loss_vec = comp[in_list, in_list].sum(axis=1)
            
            # gain_base[j_idx] = comp[j, in_set].sum()
            gain_base = comp[not_in_set, in_list].sum(axis=1)
            
            # comp_ji[j_idx, i_idx] = comp[j, i]
            comp_ji = comp[not_in_set, in_list]  # shape (len(not_in_set), m)
            
            # delta[i_idx, j_idx] = gain_base[j_idx] - comp_ji[j_idx, i_idx] - loss_vec[i_idx]
            delta = (gain_base[None, :] - loss_vec[:, None]) - comp_ji.T
            
            max_delta = delta.max(axis=1)
            best_i_idx = int(np.argmax(max_delta))
            best_delta = max_delta[best_i_idx]
            
            if best_delta > 0:
                best_j_idx = int(np.argmax(delta[best_i_idx]))
                i = in_list[best_i_idx]
                j = int(not_in_set[best_j_idx])
                
                current_indices[best_i_idx] = j
                in_set.discard(i)
                in_set.add(j)
                not_in_set = np.array([x for x in range(num_masks) if x not in in_set], dtype=np.int32)
                current_score += best_delta
                improved = True
        
        return current_indices, current_score
    
    # Strategy 1: Greedy initialization
    degrees = comp.sum(axis=1)
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    for _ in range(m):
        if time.time() > deadline - 15:
            break
        if len(selected_indices) == 0:
            scores = degrees.copy()
        else:
            scores = comp[:, selected_indices].sum(axis=1)
            scores[selected] = -1
        best = int(np.argmax(scores))
        selected[best] = True
        selected_indices.append(best)
    
    idx1, score1 = hill_climb(selected_indices, deadline - 5)
    
    # Strategy 2: Start from seed
    seed_indices = list(range(m))
    idx2, score2 = hill_climb(seed_indices, deadline - 5)
    
    best_indices = idx1 if score1 >= score2 else idx2
    best_score = max(score1, score2)
    
    # Random restarts
    rng = np.random.default_rng(42)
    num_restarts = 0
    while time.time() < deadline - 10 and num_restarts < 20:
        rand_indices = rng.choice(num_masks, size=m, replace=False)
        idx, score = hill_climb(rand_indices, deadline - 5)
        if score > best_score:
            best_score = score
            best_indices = idx
        num_restarts += 1
    
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
