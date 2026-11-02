"""
Kissing number in dimension 11: construct nonzero integer vectors in R^11
such that min pairwise distance >= max norm.
"""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # --- Phase 1: Generate candidate pool ---
    # Use weight-4 vectors (4 nonzero coords, each ±1). Norm² = 4.
    # Condition for two vectors v, w with norm² = 4:
    #   ||v - w||² >= 4  =>  8 - 2(v·w) >= 4  =>  v·w <= 2
    #
    # For weight-4 vectors, v·w ranges from -4 to 4.
    # We need v·w <= 2.
    
    # Generate all weight-4 vectors
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)  # 5280
    
    # Precompute Gram matrix
    G = candidates @ candidates.T
    
    # Conflict: G[i,j] > 2 for i != j
    # We want max independent set in the graph where edge (i,j) exists if G[i,j] > 2
    
    # --- Phase 2: Greedy with multiple orderings + local search ---
    
    # First, try a structured greedy:
    # Order by (first_nonzero, lex) to exploit symmetry-breaking
    
    # Compute a sort key: for each candidate, use its first nonzero index, then lex
    def sort_key(idx):
        v = candidates[idx]
        nonzero = np.nonzero(v)[0]
        # Sort by first nonzero, then the vector itself lex
        return (nonzero[0], tuple(v[nonzero]))
    
    # For efficiency, precompute sort keys
    keys = []
    for i in range(n_cand):
        v = candidates[i]
        nz = np.nonzero(v)[0]
        keys.append((nz[0], tuple(v[nz].tolist())))
    
    sorted_indices = np.argsort(range(n_cand), key=lambda i: keys[i])
    
    best_selected = []
    best_k = 0
    rng = np.random.default_rng(42)
    
    # Try structured ordering first
    def greedy_with_order(order):
        selected = []
        for i in order:
            if not selected:
                selected.append(i)
            else:
                if np.all(G[i, selected] <= 2):
                    selected.append(i)
        return selected
    
    # Structured pass
    sel = greedy_with_order(sorted_indices)
    best_k = len(sel)
    best_selected = sel.copy()
    
    # Multiple random restarts
    n_restarts = 300
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        perm = rng.permutation(n_cand)
        sel = greedy_with_order(perm)
        if len(sel) > best_k:
            best_k = len(sel)
            best_selected = sel.copy()
    
    # --- Phase 3: Local search (swap improvement) ---
    # Try to improve by removing one selected and adding two non-selected
    # that are compatible with all remaining selected.
    
    if time.time() < deadline - 15:
        selected_set = set(best_selected)
        available = [i for i in range(n_cand) if i not in selected_set]
        
        # Precompute conflict with selected for each available
        # For each available i, count conflicts with selected
        # And find pairs of available that are mutually compatible and both compatible with selected
        
        # This is expensive, so do it iteratively
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 10:
            iter_count += 1
            improved = False
            sel_list = list(best_selected)
            sel_set = set(sel_list)
            
            for pos, i in enumerate(sel_list):
                if time.time() > deadline - 10:
                    break
                # Remove i from selected
                temp_sel = sel_list[:pos] + sel_list[pos+1:]
                temp_set = set(temp_sel)
                
                # Find candidates compatible with temp_sel
                # For each available j not in temp_set, check if G[j, temp_sel] <= 2
                # Then find pairs (j, k) both compatible with temp_sel and with each other
                
                # This is O(N^2) per removal, which is too slow for 5000 candidates
                # Instead, precompute for each i which candidates are compatible with (all_selected minus i)
                
                # Alternative: just try random pairs
                n_tries = 50
                for _ in range(n_tries):
                    if time.time() > deadline - 10:
                        break
                    # Pick two random available
                    if len(available) < 2:
                        break
                    j, k = rng.choice(len(available), size=2, replace=False)
                    j = available[j]
                    k = available[k]
                    
                    # Check j and k compatible with temp_sel
                    if not np.all(G[j, temp_sel] <= 2):
                        continue
                    if not np.all(G[k, temp_sel] <= 2):
                        continue
                    # Check j and k compatible with each other
                    if G[j, k] > 2:
                        continue
                    
                    # Swap: remove i, add j and k
                    new_sel = temp_sel + [j, k]
                    new_sel_set = set(new_sel)
                    
                    # Verify no conflicts in new_sel
                    # (Already checked j,k compatible with temp_sel and each other,
                    #  and temp_sel was already valid)
                    
                    best_selected = new_sel
                    best_k = len(new_sel)
                    selected_set = new_sel_set
                    available = [x for x in range(n_cand) if x not in new_sel_set]
                    improved = True
                    break  # Restart scan after improvement
    
    # --- Phase 4: Try to add more via targeted search ---
    if time.time() < deadline - 5:
        sel_set = set(best_selected)
        for i in range(n_cand):
            if time.time() > deadline - 5:
                break
            if i in sel_set:
                continue
            if np.all(G[i, best_selected] <= 2):
                best_selected.append(i)
                sel_set.add(i)
                best_k += 1
    
    # Convert to final vectors
    if not best_selected:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
    
    final_vecs = candidates[best_selected]
    return final_vecs

# EVOLVE-BLOCK-END
