def _maxsat_local_search(blocks, v, k, t, deadline):
    """
    Local search: 
    1. Remove redundant blocks.
    2. Try to replace blocks that have unique coverage with new blocks that 
       cover those same unique subsets (and hopefully more), enabling further 
       optimization.
    """
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    rng = random.Random(123)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove any redundant blocks (coverage count <= 1 for all its subsets? No, 
        #    a block is redundant if NO subset it covers has count 1.
        #    i.e., for all ts in block, coverage_count[ts] > 1)
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            is_redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        
        if improved:
            continue

        # 2. Try to improve by replacing a block
        #    Pick a block with the "least" unique coverage? Or just random?
        #    Let's pick a random block and try to replace it.
        #    Actually, targeting blocks with *some* unique coverage is key.
        #    If a block has 0 unique coverage, it was already removed in step 1.
        #    So all remaining blocks have >= 1 unique subset.
        
        # Try a few random replacements
        n_blocks = len(blocks)
        if n_blocks == 0:
            break
            
        # Shuffle indices to try different blocks
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for bi in indices:
            if time.time() >= deadline:
                break
                
            old_block = blocks[bi]
            # Find unique subsets of this block
            unique_subs = []
            for ts in itertools.combinations(old_block, t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) == 1:
                    unique_subs.append(ts)
            
            if not unique_subs:
                continue # Should have been removed, but safety

            # Try to find a new block that covers all unique_subs
            # Strategy: Pick one unique_sub, say us.
            # The new block MUST contain us.
            # We can try to construct a block containing us.
            # To be a good replacement, it should cover as many other subsets as possible,
            # but the hard constraint is covering ALL unique_subs.
            
            # Heuristic: 
            # If len(unique_subs) is small, we can try to satisfy them.
            # If len(unique_subs) is large, it's hard to find a single block covering all.
            # In that case, maybe we shouldn't replace, or we should try to split?
            # But we can't add blocks (we want to minimize count).
            # So we can only replace if a single block can cover all unique_subs.
            
            # Check if union of unique_subs fits in k
            union_set = set()
            for us in unique_subs:
                union_set.update(us)
            
            if len(union_set) > k:
                continue # Cannot cover all unique subsets with one block
            
            # Construct candidate block:
            # It must contain union_set.
            # Fill remaining k - len(union_set) points randomly from the rest.
            remaining_points = set(range(v)) - union_set
            extra_needed = k - len(union_set)
            if extra_needed > len(remaining_points):
                continue # Should not happen if union_set fits in v
            
            # Try a few random completions
            found_replacement = False
            for _ in range(10): # Try 10 random completions
                if time.time() >= deadline:
                    break
                if extra_needed == 0:
                    new_block = sorted(union_set)
                else:
                    extras = rng.sample(sorted(remaining_points), extra_needed)
                    new_block = sorted(list(union_set) + extras)
                
                # Check if this new block covers all unique_subs
                # (It does by construction, since it contains union_set)
                # But we also need to ensure it doesn't break coverage of other subsets?
                # No, replacing B_old with B_new:
                # - Subsets covered by B_old but not B_new: 
                #   These are subsets of B_old that are NOT in B_new.
                #   We MUST ensure these are covered by other blocks.
                #   i.e., for any ts in B_old, if ts not in B_new, coverage_count[ts] must be > 1.
                
                valid = True
                new_block_set = set(new_block)
                for ts in itertools.combinations(old_block, t):
                    ts = tuple(sorted(ts))
                    if ts not in new_block_set:
                        if coverage_count.get(ts, 0) <= 1:
                            valid = False
                            break
                if not valid:
                    continue
                
                # If valid, check if it's "better".
                # Better means: it covers more "critical" subsets?
                # Or simply: accept it and see if it enables removals later.
                # For now, accept any valid replacement that is different.
                if new_block != old_block:
                    # Update coverage
                    # Remove old block's contribution
                    for ts in itertools.combinations(old_block, t):
                        ts = tuple(sorted(ts))
                        coverage_count[ts] -= 1
                        if coverage_count[ts] <= 0:
                            del coverage_count[ts]
                    
                    # Add new block's contribution
                    for ts in itertools.combinations(new_block, t):
                        ts = tuple(sorted(ts))
                        coverage_count[ts] = coverage_count.get(ts, 0) + 1
                    
                    blocks[bi] = new_block
                    improved = True
                    found_replacement = True
                    break
            
            if found_replacement:
                break

    return blocks
