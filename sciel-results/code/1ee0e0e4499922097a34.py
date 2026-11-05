<<<<<<< SEARCH
def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol
        
    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True
            
    return sol
=======
def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol
        
    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        # Sort by overlap with other blocks (least overlap first) to maximize pruning
        overlap_scores = []
        for idx in sol:
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                overlap_scores.append((0, idx))
                continue
            # Count how many of this block's triples are also covered by other blocks
            redundant_triples = np.sum(counts[valid_tris] > 1)
            overlap_scores.append((redundant_triples, idx))
        
        # Process in order of highest redundancy (most overlap = safest to remove)
        overlap_scores.sort(key=lambda x: -x[0])
        
        for _, idx in overlap_scores:
            if idx not in sol_set:
                continue
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True
            
    return sol
>>>>>>> REPLACE

