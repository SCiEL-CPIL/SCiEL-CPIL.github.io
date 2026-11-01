<<<<<<< SEARCH
def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start greedy
    # We don't know how many restarts we can afford. 
    # One greedy run takes maybe 0.5-2 seconds?
    # Let's try to do as many as possible.
    
    while time.time() < deadline:
        start_time = time.time()
        
        # 1. Greedy Construction
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            # Pick a random uncovered t-set to seed a new block
            # This randomization helps escape deterministic local optima
            seed_tset = random.choice(list(uncovered))
            block = list(seed_tset)
            
            while len(block) < k and time.time() < deadline - 5:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                best_candidates = []
                
                # For each candidate x not in block
                for x in range(v):
                    if x in block_set:
                        continue
                    # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = frozenset(list(R) + [x])
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_x == -1 and not best_candidates:
                    # No point adds coverage? Or gain is 0?
                    # If best_gain is 0, we should still add a point to complete the block
                    # to ensure we form a valid k-subset.
                    if best_gain == 0:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = random.choice(remaining)
                        else:
                            break
                    else:
                        break
                else:
                    # Pick randomly among best candidates to diversify
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                block_frozen = frozenset(block)
                
                # Remove all covered t-sets
                covered = set()
                for c in itertools.combinations(block, t):
                    covered.add(frozenset(c))
                uncovered -= covered
                blocks.append(block)
        
        # Check if we have full coverage
        if uncovered:
            # If time ran out during construction, we might have an incomplete covering.
            # We need to fill it.
            # Simple fill: for each uncovered t-set, add a block containing it.
            for ts in list(uncovered):
                if time.time() >= deadline - 2:
                    break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
        
        if uncovered:
            # Still uncovered? This shouldn't happen if we filled correctly, 
            # but if we broke out early, it might.
            # In that case, this run is invalid. Skip pruning and update.
            continue
            
        # 2. Local Search / Pruning
        # We have a valid covering. Try to reduce size.
        pruned_blocks = _prune_redundant_blocks(blocks, v, t)
        
        # We can try a few iterations of prune? 
        # Prune is idempotent in the sense that if no block is redundant, 
        # removing none changes nothing. 
        # But removing one block might make another redundant.
        # So we loop.
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 1:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t)
            
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
            
        # If we are very close to deadline, break to ensure return
        if time.time() > deadline - 10:
            break

    if best_blocks is None:
        # Fallback: return a valid covering if we haven't found one?
        # The loop above should have found at least one if it ran.
        # If it didn't run or failed, we need a valid result.
        # Let's do one deterministic run just in case?
        # Or return the last `blocks` if it was valid?
        # `blocks` from the last iteration might be valid but not pruned.
        # Let's check the last `blocks` variable scope. It's inside the while loop.
        # We need to keep a backup of the last valid covering.
        
        # Let's restructure to keep `last_valid_blocks`.
        # For now, if best_blocks is None, it's a problem.
        # But the loop runs at least once.
        # If the first run completes, best_blocks is set.
        # If the first run is interrupted, `uncovered` is non-empty, so `continue` is hit.
        # So `best_blocks` might remain None if the first run is always interrupted.
        # To be safe, let's force a quick valid construction if best_blocks is None.
        
        # Simple greedy without pruning, just to get a valid set.
        uncovered = set(all_tsets)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            seed_tset = next(iter(uncovered))
            block = list(seed_tset)
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set: continue
                    gain = sum(1 for R in itertools.combinations(block, t-1) if frozenset(list(R)+[x]) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem: best_x = rem[0]
                    else: break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                covered = set(frozenset(c) for c in itertools.combinations(block, t))
                uncovered -= covered
                blocks.append(block)
        
        if not uncovered:
            best_blocks = blocks
        else:
            # Emergency fill
            for ts in list(uncovered):
                if time.time() >= deadline - 0.5: break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v-1)
                    if x not in block: block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
            best_blocks = blocks

    return best_blocks
=======
def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    all_tsets_set = set(all_tsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start greedy
    while time.time() < deadline:
        start_time = time.time()
        
        # 1. Greedy Construction
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 15:
            # Pick a random uncovered t-set to seed a new block
            seed_tset = random.choice(list(uncovered))
            block = list(seed_tset)
            
            while len(block) < k and time.time() < deadline - 15:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                # For each candidate x not in block
                for x in range(v):
                    if x in block_set:
                        continue
                    # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = frozenset(list(R) + [x])
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    if best_gain == 0:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = random.choice(remaining)
                        else:
                            break
                    else:
                        break
                else:
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                covered = set()
                for c in itertools.combinations(block, t):
                    covered.add(frozenset(c))
                uncovered -= covered
                blocks.append(block)
        
        # Check if we have full coverage
        if uncovered:
            for ts in list(uncovered):
                if time.time() >= deadline - 10:
                    break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
        
        if uncovered:
            continue
            
        # 2. Prune
        pruned_blocks = _prune_redundant_blocks(blocks, v, t)
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 5:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t)
        
        # 3. Local search
        if time.time() < deadline - 5:
            ls_blocks = _local_search(pruned_blocks, v, k, t, deadline - 2)
            if len(ls_blocks) < len(pruned_blocks):
                pruned_blocks = ls_blocks
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = list(pruned_blocks)
            
        # If we are very close to deadline, break to ensure return
        if time.time() > deadline - 10:
            break

    if best_blocks is None:
        # Fallback: quick valid construction
        uncovered = set(all_tsets)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            seed_tset = next(iter(uncovered))
            block = list(seed_tset)
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set: continue
                    gain = sum(1 for R in itertools.combinations(block, t-1) if frozenset(list(R)+[x]) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem: best_x = rem[0]
                    else: break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                covered = set(frozenset(c) for c in itertools.combinations(block, t))
                uncovered -= covered
                blocks.append(block)
        
        if not uncovered:
            best_blocks = blocks
        else:
            for ts in list(uncovered):
                if time.time() >= deadline - 0.5: break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v-1)
                    if x not in block: block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
            best_blocks = blocks

    return best_blocks
>>>>>>> REPLACE
