<<<<<<< SEARCH
def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    # State: list of block indices
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    # Verify initial validity
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks
        changed = True
        while changed:
            changed = False
            # Iterate backwards to safely delete
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break # Restart loop to check from beginning again, as counts changed
        
        if not current_blocks:
            break
            
        # Phase 2: Try a swap
        pos = rng.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[pos]
        
        # Determine which triples become uncovered if we remove b_out
        uncovered_triples = []
        for idx in block_triple_indices[b_out]:
            if triple_counts[idx] == 1:
                uncovered_triples.append(idx)
                
        if not uncovered_triples:
            # Redundant, delete (should have been caught in Phase 1, but safe to handle)
            del current_blocks[pos]
            for idx in block_triple_indices[b_out]:
                triple_counts[idx] -= 1
            continue

        # Find a replacement block that covers all uncovered_triples
        if uncovered_triples:
            # Start with the triple with fewest covering blocks
            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            # Filter out blocks already in current_blocks
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                b_in = rng.choice(list(candidates))
                
                # Perform swap
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                    
                current_blocks.append(b_in)
                for idx in block_triple_indices[b_in]:
                    triple_counts[idx] += 1
                # Note: We don't update best_blocks here immediately, 
                # but the next Phase 1 might reduce it further.
                # We update best_blocks at the start of the next loop iteration.

    return best_blocks
=======
def _local_search(v, k, t, initial_blocks, block_triple_indices, block_triple_indices_np, 
                  triple_to_blocks, triple_to_blocks_np, triple_constraint,
                  full_mask, n_triples, n_blocks, deadline):
    # State: list of block indices
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts using numpy
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for b in current_blocks:
        triple_counts[block_triple_indices_np[b]] += 1
            
    # Verify initial validity
    num_uncovered = np.sum(triple_counts == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks (vectorized check)
        changed = True
        while changed:
            changed = False
            # Check each block for redundancy
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                tri_indices = block_triple_indices_np[b_out]
                # Redundant if all triples in this block have count > 1
                if np.all(triple_counts[tri_indices] > 1):
                    del current_blocks[i]
                    triple_counts[tri_indices] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break  # Restart to re-check from beginning
        
        if not current_blocks:
            break
            
        # Phase 2: Try swaps (remove one block, add another that covers exposed triples)
        pos = rng.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[pos]
        tri_indices_out = block_triple_indices_np[b_out]
        
        # Determine which triples become uncovered if we remove b_out
        exposed_mask = triple_counts[tri_indices_out] == 1
        if not np.any(exposed_mask):
            # Redundant, delete
            del current_blocks[pos]
            triple_counts[tri_indices_out] -= 1
            continue
        
        exposed_triples = tri_indices_out[exposed_mask]
        
        # Find candidate blocks that cover all exposed triples
        # Start with the most constrained exposed triple
        min_tri = exposed_triples[np.argmin(triple_constraint[exposed_triples])]
        candidates = triple_to_blocks_np[min_tri]
        
        # Intersect with blocks covering other exposed triples
        for tri in exposed_triples:
            if tri == min_tri:
                continue
            if len(candidates) == 0:
                break
            candidates = np.intersect1d(candidates, triple_to_blocks_np[tri])
        
        if len(candidates) == 0:
            continue
        
        # Filter out blocks already in current_blocks
        current_set = set(current_blocks)
        valid_candidates = [b for b in candidates if b not in current_set]
        
        if not valid_candidates:
            continue
        
        b_in = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
        
        # Perform swap
        del current_blocks[pos]
        triple_counts[tri_indices_out] -= 1
        
        current_blocks.append(b_in)
        triple_counts[block_triple_indices_np[b_in]] += 1
        
        # Immediately try to reduce after swap
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_chk = current_blocks[i]
                tri_chk = block_triple_indices_np[b_chk]
                if np.all(triple_counts[tri_chk] > 1):
                    del current_blocks[i]
                    triple_counts[tri_chk] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
    
    return best_blocks
>>>>>>> REPLACE

