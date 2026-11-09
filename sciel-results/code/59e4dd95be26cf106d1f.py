"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def _count_pairs_np(comp, selected_indices):
    """Count comparable pairs among selected indices."""
    if len(selected_indices) < 2:
        return 0
    sub = comp[np.ix_(selected_indices, selected_indices)]
    return int(sub.sum() // 2)

def _greedy_init(comp, m, num_masks, rng, strategy):
    """Greedy initialization with different strategies."""
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    if strategy == 0:
        # Pure greedy
        for _ in range(m):
            if len(selected_indices) == 0:
                scores = comp.sum(axis=1).astype(np.int32)
                best_idx = int(np.argmax(scores))
            else:
                gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                gains[selected] = -1
                best_idx = int(np.argmax(gains))
            selected_indices.append(best_idx)
            selected[best_idx] = True
            
    elif strategy == 1:
        # Greedy with noise
        scores = comp.sum(axis=1).astype(np.int32)
        for _ in range(m):
            if len(selected_indices) == 0:
                best_idx = int(np.argmax(scores))
            else:
                gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                gains[selected] = -1
                noise = rng.random(num_masks) * 1.0
                gains_noisy = gains + noise
                gains_noisy[selected] = -1
                best_idx = int(np.argmax(gains_noisy))
            selected_indices.append(best_idx)
            selected[best_idx] = True
            
    elif strategy == 2:
        # Chain-based: build multiple chains
        # Start with random elements, extend chains
        chains = []
        used = set()
        # Pick m random starting points
        candidates = list(range(num_masks))
        rng.shuffle(candidates)
        for start in candidates:
            if len(used) >= m:
                break
            if start in used:
                continue
            used.add(start)
            chain = [start]
            # Extend chain upward
            current = start
            while True:
                # Find superset of current not used
                sup_mask = comp[current, :]  # 1 where comparable
                # We want strict supersets: comp[current, j] = 1 and j != current
                # Actually comp is symmetric comparability, need to be more careful
                # Let's just pick random comparable element
                comp_candidates = np.where(sup_mask & ~np.array([1 if i in used else 0 for i in range(num_masks)]))[0]
                if len(comp_candidates) == 0:
                    break
                # Prefer larger sets (more bits)
                bit_counts = np.array([bin(i).count('1') for i in comp_candidates])
                # Weighted random
                weights = bit_counts.astype(float) + 1
                weights /= weights.sum()
                idx = int(rng.choice(len(comp_candidates), p=weights))
                nxt = int(comp_candidates[idx])
                if nxt in used:
                    break
                used.add(nxt)
                chain.append(nxt)
                current = nxt
                if len(used) >= m:
                    break
            chains.append(chain)
        
        selected_indices = []
        for c in chains:
            for idx in c:
                if idx not in used or True:  # already in used
                    selected_indices.append(idx)
                    if len(selected_indices) >= m:
                        break
            if len(selected_indices) >= m:
                break
        
        # Fill remaining with greedy
        while len(selected_indices) < m:
            sel_set = set(selected_indices)
            gains = comp.sum(axis=1).astype(np.int32)
            # Subtract already selected
            for idx in selected_indices:
                gains -= comp[:, idx]
            gains[np.array(selected_indices, dtype=int)] = -1
            best_idx = int(np.argmax(gains))
            selected_indices.append(best_idx)
        
        for idx in selected_indices:
            selected[idx] = True
            
    else:
        # Random
        candidates = np.arange(num_masks)
        rng.shuffle(candidates)
        for idx in candidates[:m]:
            selected_indices.append(int(idx))
            selected[int(idx)] = True
    
    return selected_indices, selected

def _single_swap_ls(comp, in_set, score_in_set, selected_indices, m, num_masks, deadline):
    """Single-swap local search."""
    current_score = int(score_in_set[list(selected_indices)].sum() // 2)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        masked_scores = np.where(in_set, -1, score_in_set)
        max_val = int(np.max(masked_scores))
        max_idx = int(np.argmax(masked_scores))
        
        if max_val < 0:
            break
        
        # Find min score in set
        min_score = np.inf
        min_idx = -1
        for A in selected_indices:
            if score_in_set[A] < min_score:
                min_score = score_in_set[A]
                min_idx = A
        
        best_delta = max_val - min_score
        if best_delta <= 0:
            break
        
        # Apply swap
        score_in_set += comp[:, max_idx] - comp[:, min_idx]
        selected_indices[selected_indices.index(min_idx)] = max_idx
        in_set[min_idx] = False
        in_set[max_idx] = True
        current_score += best_delta
        improved = True
    
    return current_score

def _two_swap_ls(comp, in_set, score_in_set, selected_indices, m, num_masks, deadline):
    """Vectorized 2-swap local search."""
    current_score = int(score_in_set[list(selected_indices)].sum() // 2)
    
    # Precompute: for each pair (A, B) in set, and each C not in set,
    # the "base" contribution if we remove A and add C
    # delta = (sC - cCA - sA) + best over (B, D) of [2*cAB + (sD - cDA - sB) + cCD - cCB - cDB]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta_2 = 0
        best_A = -1
        best_B = -1
        best_C = -1
        best_D = -1
        
        sel_list = list(selected_indices)
        
        # For each A in set
        for A in sel_list:
            if time.time() > deadline:
                break
            
            c_A = comp[A, :]
            s_A = score_in_set[A]
            
            # Potential for C: sC - cCA
            potential_C = score_in_set - c_A
            potential_C[in_set] = -1000
            
            # Get top K candidates for C
            K = 30
            top_C = np.argsort(potential_C)[-K:][::-1]
            top_C = [int(c) for c in top_C if potential_C[c] > -500]
            
            if not top_C:
                continue
            
            for C in top_C:
                if time.time() > deadline:
                    break
                
                c_CA = comp[C, A]
                constant = score_in_set[C] - c_CA - s_A
                
                # For each B in S\{A}, compute best D
                # For fixed A, C, B:
                # term = constant + 2*cAB + (sD - cDA - sB) + cCD - cCB - cDB
                # = (constant + 2*cAB - cCB) + (sD - cDA - cDB + cCD) - sB
                # Let val_D = sD - cDA - cDB + cCD
                # Then total = constant + 2*cAB - cCB + val_D - sB
                
                # Vectorize over B: for all B in S\{A}, compute base_B = constant + 2*cAB - cCB
                # Then for each B, find best D maximizing val_D - sB
                
                B_list = [b for b in sel_list if b != A]
                if not B_list:
                    continue
                
                B_arr = np.array(B_list, dtype=int)
                c_AB = comp[A, B_arr]  # shape (len(B),)
                c_CB = comp[C, B_arr]  # shape (len(B),)
                s_B = score_in_set[B_arr]  # shape (len(B),)
                
                # base_B[i] = constant + 2*c_AB[i] - c_CB[i]
                base_B = constant + 2 * c_AB - c_CB
                
                # For each B, compute val_D = sD - cDA - cDB + cCD
                # val_D[:, i] = score_in_set - comp[:, A] - comp[:, B_arr[i]] + comp[C, :]
                # Shape: (num_masks, len(B))
                
                # This is large: num_masks * len(B) = 4096 * 39 ~ 160K, times K=30 C's, times m=40 A's
                # = 4096 * 39 * 30 * 40 = 191M operations... too slow
                
                # Instead, for each B, find best D separately but vectorized over D
                # For a given B: val_D = score_in_set - comp[:, A] - comp[:, B] + comp[C, :]
                # best_D = argmax of val_D where not in_set and D != C
                
                # Process B's in batches to keep memory reasonable
                batch_size = 10
                for b_start in range(0, len(B_arr), batch_size):
                    if time.time() > deadline:
                        break
                    b_end = min(b_start + batch_size, len(B_arr))
                    B_batch = B_arr[b_start:b_end]
                    base_batch = base_B[b_start:b_end]
                    s_B_batch = s_B[b_start:b_end]
                    
                    # val_D shape: (num_masks, batch_size)
                    val_D = (score_in_set[:, None] - comp[:, A][:, None] - 
                             comp[:, B_batch] + comp[C, :][:, None])
                    
                    # Mask: D not in set, D != C
                    mask_D = (~in_set) & (np.arange(num_masks) != C)
                    val_D[~mask_D, :] = -1000
                    
                    # best_D per B in batch
                    best_D_batch = np.argmax(val_D, axis=0)
                    best_val_D_batch = val_D[best_D_batch, np.arange(len(B_batch))]
                    
                    # total_delta per B = base_batch + best_val_D_batch - s_B_batch
                    total_batch = base_batch + best_val_D_batch - s_B_batch
                    
                    for bi in range(len(B_batch)):
                        if total_batch[bi] > best_delta_2:
                            best_delta_2 = int(total_batch[bi])
                            best_A = A
                            best_B = int(B_batch[bi])
                            best_C = C
                            best_D = int(best_D_batch[bi])
        
        if best_delta_2 <= 0:
            break
        
        # Apply 2-swap
        score_in_set += comp[:, best_C] + comp[:, best_D] - comp[:, best_A] - comp[:, best_B]
        
        idx_A = selected_indices.index(best_A)
        idx_B = selected_indices.index(best_B)
        selected_indices[idx_A] = best_C
        selected_indices[idx_B] = best_D
        
        in_set[best_A] = False
        in_set[best_B] = False
        in_set[best_C] = True
        in_set[best_D] = True
        
        current_score += best_delta_2
        improved = True
    
    return current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    masks = np.arange(num_masks, dtype=np.int32)
    
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    best_overall_score = 0
    best_overall_indices = None
    
    rng = np.random.RandomState(42)
    
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # 2. Greedy Initialization
        strategy = restart % 4
        selected_indices, selected = _greedy_init(comp, m, num_masks, rng, strategy)
        
        # 3. Local Search
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
            
        score_in_set = comp[:, list(selected_indices)].sum(axis=1).astype(np.int32)
        
        # Single-swap local search
        current_score = _single_swap_ls(comp, in_set, score_in_set, selected_indices, m, num_masks, deadline - 15)
        
        # 2-swap local search
        current_score = _two_swap_ls(comp, in_set, score_in_set, selected_indices, m, num_masks, deadline - 10)
        
        # Final single-swap polish
        current_score = _single_swap_ls(comp, in_set, score_in_set, selected_indices, m, num_masks, deadline - 5)
        
        # Update best
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_indices = list(selected_indices)
    
    # Convert to list of lists of Python ints
    if best_overall_indices is None:
        result = []
        for mask in selected_indices:
            mask_int = int(mask)
            subset = [b for b in range(n) if (mask_int >> b) & 1]
            result.append(subset)
        return result
    
    result = []
    for mask in best_overall_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
