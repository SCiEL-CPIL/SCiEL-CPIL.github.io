"""Improved program for kissing number in dimension 11."""
import time
import numpy as np

D = 11
DEADLINE = time.time() + 130  # safety margin within 150s limit


def construct():
    start = time.time()
    
    # Generate all candidate vectors with up to 4 non-zero entries
    # Norms: 1, 2, 3, 4 (squared)
    # We'll focus on vectors with 2 and 3 non-zero entries first,
    # as they give the best tradeoff between count and sparsity.
    
    from itertools import combinations, product
    
    candidates = []
    seen = set()
    
    # Vectors with 2 non-zero entries: norm^2 = 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                key = tuple(v)
                if key not in seen:
                    seen.add(key)
                    candidates.append(v)
    
    # Vectors with 3 non-zero entries: norm^2 = 3
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            v[k] = sk
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                candidates.append(v)
    
    # Vectors with 1 non-zero entry: norm^2 = 1
    for i in range(D):
        for si in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                candidates.append(v)
    
    n_candidates = len(candidates)
    cand_array = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(cand_array ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)  # = 3
    
    # Build incompatibility: two vectors are incompatible if ||vi - vj||^2 < max_norm_sq
    # ||vi - vj||^2 = ||vi||^2 + ||vj||^2 - 2*vi.vj
    # So incompatible if: norms_sq[i] + norms_sq[j] - 2*dot(i,j) < max_norm_sq
    # i.e., dot(i,j) > (norms_sq[i] + norms_sq[j] - max_norm_sq) / 2
    
    # For greedy, we'll track selected vectors and their norms
    # Use a smarter approach: sort candidates by degree (fewest conflicts first)
    
    # Precompute dot products in batches to save memory
    # Actually, with ~1500 candidates, a full matrix is fine
    # dot_matrix[i,j] = candidates[i] . candidates[j]
    
    if n_candidates > 3000:
        # Memory concern; use streaming
        dot_products = np.zeros((n_candidates, n_candidates), dtype=np.int32)
        for i in range(n_candidates):
            dot_products[i, i:] = cand_array[i] @ cand_array[i:].T
            dot_products[i:, i] = dot_products[i, i:]
    else:
        dot_products = cand_array @ cand_array.T
    
    # Incompatibility matrix: True if pair is incompatible
    # Incompatible if norms_sq[i] + norms_sq[j] - 2*dot(i,j) < max_norm_sq
    # i.e., 2*dot(i,j) > norms_sq[i] + norms_sq[j] - max_norm_sq
    # Since all are integers, we can use integer arithmetic
    
    # For efficiency, compute threshold matrix
    # thresh[i,j] = norms_sq[i] + norms_sq[j] - max_norm_sq
    # Incompatible if 2*dot(i,j) > thresh[i,j]
    
    thresh = norms_sq[:, None] + norms_sq[None, :] - max_norm_sq
    
    # Incompatible where 2*dot > thresh AND i != j
    incompatible = (2 * dot_products > thresh) & (np.eye(n_candidates, dtype=bool))
    
    # Compute degree (number of incompatibilities) for each candidate
    degrees = incompatible.sum(axis=1)
    
    # Greedy: pick vertices in order of increasing degree
    order = np.argsort(degrees)
    
    selected = []
    selected_set = set()
    # Track which candidates are still "available" (not incompatible with any selected)
    available = np.ones(n_candidates, dtype=bool)
    
    for idx in order:
        if not available[idx]:
            continue
        # Check if idx is compatible with all already selected
        # Since we maintain available, idx is compatible if available[idx] is True
        # But we need to update available for all future candidates
        selected.append(idx)
        selected_set.add(idx)
        
        # Update available: candidates incompatible with idx become unavailable
        # incompatible[idx] gives all j incompatible with idx
        available[incompatible[idx]] = False
        available[idx] = False  # mark as used
    
    # Now try to add more vectors via local search
    # Try to swap: remove one selected, add two non-selected if compatible
    current_size = len(selected)
    
    # Build a faster compatibility check using the incompatible matrix
    # For local search, we'll try random swaps for a while
    
    best_selected = set(selected)
    best_size = current_size
    
    # Local search: try to improve by swapping
    # Represent selected as a boolean array for fast operations
    sel_mask = np.zeros(n_candidates, dtype=bool)
    for idx in selected:
        sel_mask[idx] = True
    
    # Number of non-selected compatible with all selected
    # A candidate j is addable if for all i in selected, not incompatible[i,j]
    # i.e., incompatible[sel_mask, j].all() is False... no, we need NO incompatibilities
    # Addable if (~incompatible[sel_mask, :]).all(axis=0)
    
    # Precompute: for each j, count how many selected are incompatible with j
    # incomp_count[j] = incompatible[sel_mask, j].sum()
    # Addable if incomp_count[j] == 0
    
    if len(selected) > 0:
        incomp_count = incompatible[sel_mask, :].sum(axis=0)
    else:
        incomp_count = np.zeros(n_candidates, dtype=np.int32)
    
    # Try to add any addable non-selected vectors
    changed = True
    while changed:
        changed = False
        non_sel = ~sel_mask
        addable = (incomp_count == 0) & non_sel
        for j in np.where(addable)[0]:
            sel_mask[j] = True
            # Update incomp_count for all other non-selected
            # incomp_count += incompatible[j, non_sel]... but need to be careful
            # For all k not selected, incomp_count[k] += incompatible[j, k]
            # But only for k where not incompatible... no, we just add
            incomp_count[~sel_mask] += incompatible[j, ~sel_mask]
            changed = True
    
    current_size = sel_mask.sum()
    best_size = max(best_size, current_size)
    best_selected = set(np.where(sel_mask)[0])
    
    # Now try swap moves: remove one selected, see if we can add more
    # Try each selected vertex for removal
    sel_list = list(best_selected)
    
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for remove_idx in sel_list:
            # Try removing remove_idx
            sel_mask[remove_idx] = False
            # Recompute incomp_count
            if sel_mask.any():
                new_incomp_count = incompatible[sel_mask, :].sum(axis=0)
            else:
                new_incomp_count = np.zeros(n_candidates, dtype=np.int32)
            
            # Try to greedily add as many as possible
            non_sel = ~sel_mask
            addable = (new_incomp_count == 0) & non_sel
            added = 0
            for j in np.where(addable)[0]:
                sel_mask[j] = True
                new_incomp_count[~sel_mask] += incompatible[j, ~sel_mask]
                added += 1
            
            new_size = sel_mask.sum()
            if new_size > best_size:
                best_size = new_size
                best_selected = set(np.where(sel_mask)[0])
                sel_list = list(best_selected)
                improved = True
            else:
                # Restore
                sel_mask[remove_idx] = True
    
    # Extract the final vectors
    final_indices = np.array(list(best_selected), dtype=np.int64)
    result = cand_array[final_indices]
    
    return result
