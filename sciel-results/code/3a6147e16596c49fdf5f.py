def _local_search(blocks, v, k, t, tsub_to_idx, deadline, rng, all_tsubs):
    """Lightweight local search: random swaps to enable redundancy removal."""
    if not blocks:
        return blocks
    
    # Build initial coverage
    cover_count = Counter()
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                tsubs.append(idx)
                cover_count[idx] += 1
        block_tsubs.append(tsubs)
        
    n_blocks = len(blocks)
    
    # Try to improve by swapping points in blocks
    # We want to find a swap that allows removing a block.
    # Strategy: Pick a block B. Try to change it so that it no longer covers some unique t-subsets,
    # or covers new ones, but primarily we want to make some other block redundant.
    # Simpler: Just try random swaps and see if _remove_redundant yields fewer blocks.
    
    max_attempts = 100
    attempts = 0
    improved = True
    while improved and attempts < max_attempts and time.time() < deadline - 2:
        improved = False
        attempts += 1
        
        # Pick a random block
        i = rng.randrange(len(blocks))
        # Pick a random position in the block
        pi = rng.randrange(k)
        old_point = blocks[i][pi]
        
        # Try to swap with a random point not in the block
        candidates = [x for x in range(v) if x not in set(blocks[i])]
        if not candidates:
            continue
        new_point = rng.choice(candidates)
        
        # Perform swap
        blocks[i][pi] = new_point
        blocks[i].sort()
        
        # Check validity: all t-subsets must still be covered.
        # We only need to check t-subsets that were covered by block i and are now potentially uncovered.
        # And t-subsets that are newly covered (doesn't hurt validity, only might help redundancy).
        
        # Fast validity check:
        # 1. Identify t-subsets in the OLD block i that are NOT in the NEW block i.
        # 2. For each, check if it is covered by ANY other block.
        
        old_block_set = set(old_point) # Placeholder, need full old block
        # Let's just rebuild cover count for the specific block and check global coverage?
        # No, that's expensive.
        
        # Alternative: Just run _remove_redundant on a copy and see if it's smaller.
        # But _remove_redundant modifies in place.
        
        # To be safe and simple:
        # Check if the swap broke coverage.
        # We can pre-compute which t-subsets are covered by exactly 1 block (critical).
        # If we swap a point in block i, we might lose coverage for some critical t-subsets.
        
        # Let's do a quick check:
        # Re-calculate tsubs for the new block i.
        new_block_tsubs = set()
        for ts in itertools.combinations(blocks[i], t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                new_block_tsubs.add(idx)
        
        # Old tsubs for block i
        old_block_tsubs = set(block_tsubs[i])
        
        # Tsubs lost by block i
        lost_tsubs = old_block_tsubs - new_block_tsubs
        
        # For each lost tsub, is it covered by any other block?
        valid = True
        for idx in lost_tsubs:
            if cover_count[idx] == 1:
                # It was only covered by block i. Now it's lost.
                # Unless it's in new_block_tsubs (which it isn't by definition of lost)
                valid = False
                break
        
        if valid:
            # Update cover_count and block_tsubs
            for idx in lost_tsubs:
                cover_count[idx] -= 1
            gained_tsubs = new_block_tsubs - old_block_tsubs
            for idx in gained_tsubs:
                cover_count[idx] += 1
            block_tsubs[i] = list(new_block_tsubs)
            
            # Try to remove redundant blocks
            new_blocks = _remove_redundant([list(b) for b in blocks], v, k, t, tsub_to_idx, deadline)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                # Rebuild cover_count and block_tsubs for the new set
                cover_count = Counter()
                block_tsubs = []
                for block in blocks:
                    tsubs = []
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx.get(tuple(ts))
                        if idx is not None:
                            tsubs.append(idx)
                            cover_count[idx] += 1
                    block_tsubs.append(tsubs)
                improved = True
            else:
                # Revert swap
                blocks[i][pi] = old_point
                blocks[i].sort()
                # Revert cover_count
                for idx in lost_tsubs:
                    cover_count[idx] += 1
                for idx in gained_tsubs:
                    cover_count[idx] -= 1
                block_tsubs[i] = list(old_block_tsubs)
        else:
            # Revert swap
            blocks[i][pi] = old_point
            blocks[i].sort()

    return blocks
