<<<<<<< SEARCH
    def swap_local_search(current_blocks, coverage_count):
        """Try to replace pairs of blocks with a single block that covers all their unique t-subsets."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks, coverage_count
        
        blocks_set = set(current_blocks)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            
            n = len(current_blocks)
            tried = set()
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b1 = current_blocks[i]
                for j in range(i + 1, n):
                    b2 = current_blocks[j]
                    pair_key = (min(b1, b2), max(b1, b2))
                    if pair_key in tried:
                        continue
                    tried.add(pair_key)
                    
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]
                    
                    unique_to_1 = []
                    unique_to_2 = []
                    for sub in subs1:
                        if coverage_count[sub] == 1:
                            unique_to_1.append(sub)
                    for sub in subs2:
                        if coverage_count[sub] == 1:
                            unique_to_2.append(sub)
                    
                    if not unique_to_1 and not unique_to_2:
                        improved = True
                        coverage_count[subs1] -= 1
                        coverage_count[subs2] -= 1
                        current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        break
                    elif unique_to_1 and not unique_to_2:
                        candidates = t_to_blocks[unique_to_1[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_1):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    elif unique_to_2 and not unique_to_1:
                        candidates = t_to_blocks[unique_to_2[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_2):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    else:
                        pass
            if improved:
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                blocks_set = set(current_blocks)
        return current_blocks, coverage_count
=======
    def swap_local_search(current_blocks, coverage_count):
        """Aggressive local search: remove redundant blocks, try 2->1 and 2->2 swaps."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks, coverage_count
        
        blocks_set = set(current_blocks)
        improved = True
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            iter_count += 1
            
            n = len(current_blocks)
            
            # Phase 1: Try to remove single redundant blocks
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b = current_blocks[i]
                subs = block_t_indices[b]
                if np.all(coverage_count[subs] > 1):
                    coverage_count[subs] -= 1
                    current_blocks = [x for x in current_blocks if x != b]
                    blocks_set.discard(b)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try 2->1 swaps (replace two blocks with one)
            n = len(current_blocks)
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b1 = current_blocks[i]
                subs1 = block_t_indices[b1]
                unique_to_1 = subs1[coverage_count[subs1] == 1]
                
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit - 3.0:
                        return current_blocks, coverage_count
                    b2 = current_blocks[j]
                    subs2 = block_t_indices[b2]
                    unique_to_2 = subs2[coverage_count[subs2] == 1]
                    
                    if len(unique_to_1) == 0 and len(unique_to_2) == 0:
                        # Both blocks are redundant - remove both
                        coverage_count[subs1] -= 1
                        coverage_count[subs2] -= 1
                        current_blocks = [x for x in current_blocks if x not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        improved = True
                        break
                    elif len(unique_to_1) > 0 and len(unique_to_2) == 0:
                        # Only b1 has unique subsets - try to find a replacement for b1 that also covers b2's unique (none)
                        # Try to replace b1 with a block that covers all unique_to_1
                        if len(unique_to_1) <= 3:  # Limit search for efficiency
                            # Find candidates that cover the most unique_to_1 subsets
                            candidates = t_to_blocks[unique_to_1[0]]
                            best_cand = -1
                            best_cov = 0
                            for cand in candidates:
                                if cand == b1 or cand == b2:
                                    continue
                                cand_subs = block_t_indices[cand]
                                cov = np.sum(np.isin(cand_subs, unique_to_1))
                                if cov > best_cov:
                                    best_cov = cov
                                    best_cand = cand
                                if best_cov == len(unique_to_1):
                                    break
                            if best_cand >= 0 and best_cov == len(unique_to_1):
                                coverage_count[subs1] -= 1
                                coverage_count[subs2] -= 1
                                coverage_count[block_t_indices[best_cand]] += 1
                                current_blocks = [x for x in current_blocks if x not in (b1, b2)]
                                blocks_set.discard(b1)
                                blocks_set.discard(b2)
                                blocks_set.add(best_cand)
                                improved = True
                                break
                    elif len(unique_to_2) > 0 and len(unique_to_1) == 0:
                        # Only b2 has unique subsets
                        if len(unique_to_2) <= 3:
                            candidates = t_to_blocks[unique_to_2[0]]
                            best_cand = -1
                            best_cov = 0
                            for cand in candidates:
                                if cand == b1 or cand == b2:
                                    continue
                                cand_subs = block_t_indices[cand]
                                cov = np.sum(np.isin(cand_subs, unique_to_2))
                                if cov > best_cov:
                                    best_cov = cov
                                    best_cand = cand
                                if best_cov == len(unique_to_2):
                                    break
                            if best_cand >= 0 and best_cov == len(unique_to_2):
                                coverage_count[subs1] -= 1
                                coverage_count[subs2] -= 1
                                coverage_count[block_t_indices[best_cand]] += 1
                                current_blocks = [x for x in current_blocks if x not in (b1, b2)]
                                blocks_set.discard(b1)
                                blocks_set.discard(b2)
                                blocks_set.add(best_cand)
                                improved = True
                                break
                    else:
                        # Both have unique subsets - try 2->2 swap
                        # Find if we can replace {b1, b2} with two other blocks
                        # that cover all unique subsets from both
                        all_unique = np.concatenate([unique_to_1, unique_to_2])
                        if len(all_unique) <= 6:  # Limit for efficiency
                            # Find best pair of replacement blocks
                            # Use the most constrained unique subset to find candidates
                            candidates1 = t_to_blocks[unique_to_1[0]]
                            candidates2 = t_to_blocks[unique_to_2[0]]
                            
                            # Try to find a pair (c1, c2) where c1 covers unique_to_1 and c2 covers unique_to_2
                            # or c1 covers some of both and c2 covers the rest
                            best_pair = None
                            best_total = 0
                            
                            for c1 in list(candidates1)[:20]:  # Limit candidates
                                if c1 == b1 or c1 == b2:
                                    continue
                                c1_subs = block_t_indices[c1]
                                c1_cov_u1 = np.sum(np.isin(c1_subs, unique_to_1))
                                c1_cov_u2 = np.sum(np.isin(c1_subs, unique_to_2))
                                
                                for c2 in list(candidates2)[:20]:
                                    if c2 == b1 or c2 == b2 or c2 == c1:
                                        continue
                                    c2_subs = block_t_indices[c2]
                                    c2_cov_u1 = np.sum(np.isin(c2_subs, unique_to_1))
                                    c2_cov_u2 = np.sum(np.isin(c2_subs, unique_to_2))
                                    
                                    # Check if union of c1 and c2 covers all unique subsets
                                    covered_u1 = np.sum(np.isin(np.concatenate([c1_subs, c2_subs]), unique_to_1))
                                    covered_u2 = np.sum(np.isin(np.concatenate([c1_subs, c2_subs]), unique_to_2))
                                    
                                    if covered_u1 >= len(unique_to_1) and covered_u2 >= len(unique_to_2):
                                        total = c1_cov_u1 + c1_cov_u2 + c2_cov_u1 + c2_cov_u2
                                        if total > best_total:
                                            best_total = total
                                            best_pair = (c1, c2)
                                        break
                            
                            if best_pair is not None:
                                c1, c2 = best_pair
                                coverage_count[subs1] -= 1
                                coverage_count[subs2] -= 1
                                coverage_count[block_t_indices[c1]] += 1
                                coverage_count[block_t_indices[c2]] += 1
                                current_blocks = [x for x in current_blocks if x not in (b1, b2)]
                                blocks_set.discard(b1)
                                blocks_set.discard(b2)
                                blocks_set.add(c1)
                                blocks_set.add(c2)
                                improved = True
                                break
            
            if improved:
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                blocks_set = set(current_blocks)
        
        return current_blocks, coverage_count
>>>>>>> REPLACE

