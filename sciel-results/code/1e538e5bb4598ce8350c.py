"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 115
    
    # Pre-generate all vectors in {-1, 0, 1}^11
    n_vectors = 3 ** D
    vectors = np.zeros((n_vectors, D), dtype=np.int64)
    for i in range(n_vectors):
        val = i
        for d in range(D):
            vectors[i, d] = val % 3 - 1
            val //= 3
            
    norms_sq = np.sum(vectors * vectors, axis=1)
    nonzero_mask = norms_sq > 0
    vectors = vectors[nonzero_mask]
    norms_sq = norms_sq[nonzero_mask]
    
    best_set = []
    best_score = 0
    
    # Try different max norm thresholds N
    candidate_Ns = [2, 3, 4, 5, 6]
    
    rng = np.random.default_rng(42)
    
    for N in candidate_Ns:
        if time.time() > deadline:
            break
            
        mask = norms_sq <= N
        pool = vectors[mask]
        pool_norms = norms_sq[mask]
        n_pool = len(pool)
        
        if n_pool == 0:
            continue
        
        # We will use a greedy algorithm with local search (swapping)
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            indices = rng.permutation(n_pool)
            
            # Selected set as list of indices into pool
            selected_indices = []
            
            # Greedy Phase
            for idx in indices:
                v = pool[idx]
                v_norm = pool_norms[idx]
                
                if not selected_indices:
                    selected_indices.append(idx)
                    continue
                
                # Check compatibility with all selected
                sel_vecs = pool[selected_indices]
                sel_norms = pool_norms[selected_indices]
                
                dots = np.dot(sel_vecs, v)
                dists_sq = v_norm + sel_norms - 2.0 * dots
                
                if np.all(dists_sq >= N):
                    selected_indices.append(idx)
            
            # Local Search Phase: Try to swap one out, one in
            # We maintain a set of "conflict counts" or just re-evaluate
            improved = True
            max_swaps = 50
            swap_count = 0
            
            while improved and swap_count < max_swaps:
                improved = False
                swap_count += 1
                
                # Try to add a random unselected vector by removing a conflicting one
                unselected_indices = set(range(n_pool)) - set(selected_indices)
                if not unselected_indices:
                    break
                
                # Sample a few candidates to try
                candidates = list(unselected_indices)
                rng.shuffle(candidates)
                
                for cand_idx in candidates[:5]: # Try 5 random candidates
                    v = pool[cand_idx]
                    v_norm = pool_norms[cand_idx]
                    
                    # Find conflicts
                    sel_vecs = pool[selected_indices]
                    sel_norms = pool_norms[selected_indices]
                    
                    dots = np.dot(sel_vecs, v)
                    dists_sq = v_norm + sel_norms - 2.0 * dots
                    
                    conflict_mask = dists_sq < N
                    conflicts = np.where(conflict_mask)[0]
                    
                    if len(conflicts) == 0:
                        # Can add directly (shouldn't happen if greedy was exhaustive, but maybe due to order)
                        selected_indices.append(cand_idx)
                        improved = True
                        break
                    elif len(conflicts) == 1:
                        # Swap: remove the one conflict, add cand
                        # Check if this swap is beneficial? It keeps size same.
                        # We do it to escape local minima.
                        rem_idx = selected_indices[conflicts[0]]
                        selected_indices[conflicts[0]] = cand_idx
                        improved = True
                        break
                    elif len(conflicts) > 1:
                        # Try to remove one conflict and see if we can add cand AND another?
                        # Too complex. Just try single swaps for now.
                        pass
            
            current_score = len(selected_indices)
            
            if current_score > best_score:
                best_score = current_score
                best_set = [pool[i] for i in selected_indices]
                
    # Fallback
    if not best_set:
        mask = norms_sq == 2
        best_set = vectors[mask].tolist()
        best_score = len(best_set)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
