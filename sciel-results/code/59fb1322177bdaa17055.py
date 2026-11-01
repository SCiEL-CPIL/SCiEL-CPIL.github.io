<<<<<<< SEARCH
    def remove_redundant(blocks, time_limit):
        if not blocks:
            return []
        if time.time() > time_limit:
            return blocks
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        block_tsubs = []
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result_blocks = list(blocks)
        result_tsubs = list(block_tsubs)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                redundant = True
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in result_blocks]
=======
    def remove_redundant(blocks, time_limit):
        if not blocks:
            return []
        if time.time() > time_limit:
            return blocks
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        # Precompute t-subset indices for each block
        block_tsub_idxs = []
        for b in blocks:
            idx_list = []
            for ts in itertools.combinations(b, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    idx_list.append(idx)
                    cover_count[idx] += 1
            block_tsub_idxs.append(idx_list)
        
        result_blocks = list(blocks)
        result_tsub_idxs = list(block_tsub_idxs)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                # Check if block i is redundant: all its t-subsets are covered by other blocks
                redundant = True
                for idx in result_tsub_idxs[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for idx in result_tsub_idxs[i]:
                        cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsub_idxs.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in result_blocks]
>>>>>>> REPLACE

