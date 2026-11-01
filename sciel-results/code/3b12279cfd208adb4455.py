<<<<<<< SEARCH
    best_solution = optimize_redundancy(best_solution)
    
    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                idx = ts_index_map[ts]
                if cover_count[idx] == 1:
                    unique_ts.append(idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            for ts in itertools.combinations(old_block, t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            
            improved = True
            
        return blocks, improved
=======
    best_solution = optimize_redundancy(best_solution)
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            unique_ts = []
            b_idx = get_block_idx(blocks[i])
            if b_idx >= 0:
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
            else:
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            old_b_idx = get_block_idx(old_block)
            if old_b_idx >= 0:
                for ts_idx in block_to_ts[old_b_idx]:
                    cover_count[ts_idx] -= 1
            else:
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
            new_b_idx = get_block_idx(new_block)
            if new_b_idx >= 0:
                for ts_idx in block_to_ts[new_b_idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
            
            improved = True
            
        return blocks, improved
>>>>>>> REPLACE

