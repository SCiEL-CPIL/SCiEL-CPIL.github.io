<<<<<<< SEARCH
    def local_search_replace(blocks, time_limit):
        """Try to replace pairs of blocks with a single block."""
        n = len(blocks)
        improved = True
        while improved and time.time() < time_limit - 5:
            improved = False
            for i in range(n):
                if time.time() > time_limit - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit - 5:
                        break
                    # Find t-subsets only covered by block i or j
                    bi_set = set(blocks[i])
                    bj_set = set(blocks[j])
                    other_blocks = [blocks[m] for m in range(n) if m != i and m != j]
                    
                    required = set()
                    for ts in itertools.combinations(blocks[i], t):
                        ts_set = set(ts)
                        if not any(ts_set.issubset(set(ob)) for ob in other_blocks):
                            required.add(ts)
                    for ts in itertools.combinations(blocks[j], t):
                        ts_set = set(ts)
                        if not any(ts_set.issubset(set(ob)) for ob in other_blocks):
                            required.add(ts)
                    
                    if not required:
                        # Both blocks are redundant (should have been caught by remove_redundant)
                        continue
                    
                    req_points = set()
                    for ts in required:
                        req_points.update(ts)
                    
                    if len(req_points) > k:
                        continue
                    
                    # Try to build a block containing req_points
                    new_block = set(req_points)
                    for attempt in range(30):
                        trial = set(new_block)
                        while len(trial) < k:
                            candidates = [x for x in range(v) if x not in trial]
                            if not candidates:
                                break
                            trial.add(rng_choice(candidates, i * 1000 + j * 7 + attempt))
                        
                        # Check if trial covers all required
                        ok = all(set(ts).issubset(trial) for ts in required)
                        if ok:
                            candidate = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [sorted(trial)]
                            if is_valid(candidate):
                                blocks = candidate
                                n = len(blocks)
                                improved = True
                                break
                    if improved:
                        break
        return blocks
=======
    def local_search_replace(blocks, time_limit):
        """Try to replace pairs of blocks with a single block, using bitsets."""
        n = len(blocks)
        improved = True
        while improved and time.time() < time_limit - 5:
            improved = False
            # Precompute masks
            masks = []
            for block in blocks:
                bi = k_index.get(block)
                if bi is not None:
                    masks.append(block_masks[bi])
                else:
                    m = 0
                    for ts in itertools.combinations(block, t):
                        m |= 1 << t_index[ts]
                    masks.append(m)
            full_mask = (1 << n_t) - 1
            for i in range(n):
                if time.time() > time_limit - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit - 5:
                        break
                    # Find t-subsets only covered by block i or j (not by others)
                    other_union = 0
                    for m in range(n):
                        if m != i and m != j:
                            other_union |= masks[m]
                    # Required: t-subsets in (mask_i | mask_j) but not in other_union
                    required_mask = (masks[i] | masks[j]) & ~other_union
                    if required_mask == 0:
                        continue
                    
                    # Extract required t-subsets
                    req_points = set()
                    rm = required_mask
                    while rm:
                        bit = rm & -rm
                        ti = bit.bit_length() - 1
                        req_points.update(all_t[ti])
                        rm ^= bit
                    
                    if len(req_points) > k:
                        continue
                    
                    # Try to build a block containing req_points
                    new_block = set(req_points)
                    for attempt in range(20):
                        trial = set(new_block)
                        while len(trial) < k:
                            candidates = [x for x in range(v) if x not in trial]
                            if not candidates:
                                break
                            trial.add(rng_choice(candidates, i * 1000 + j * 7 + attempt))
                        
                        if len(trial) != k:
                            break
                        
                        # Check if trial covers all required
                        trial_bi = k_index.get(tuple(sorted(trial)))
                        if trial_bi is not None:
                            trial_mask = block_masks[trial_bi]
                        else:
                            trial_mask = 0
                            for ts in itertools.combinations(trial, t):
                                trial_mask |= 1 << t_index[ts]
                        
                        if (trial_mask & required_mask) == required_mask:
                            candidate = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [sorted(trial)]
                            if is_valid(candidate):
                                blocks = candidate
                                n = len(blocks)
                                improved = True
                                break
                    if improved:
                        break
        return blocks
>>>>>>> REPLACE

