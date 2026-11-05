<<<<<<< SEARCH
    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Count coverage
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            # Check if any triple in this block is exclusively covered by it
            # i.e., cover_count[t] == 1
            trs = block_triples_np[b_idx]
            # Vectorized check: is any triple in trs having cover_count == 1?
            has_exclusive = np.any(cover_count[trs] == 1)
            
            if not has_exclusive:
                # Remove
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count
=======
    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Count coverage
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Iterate multiple times to cascade removals
        changed = True
        while changed:
            changed = False
            new_indices_tmp = []
            new_cover_count_tmp = np.zeros(num_triples, dtype=np.int32)
            
            for b_idx in new_indices if changed else indices:
                trs = block_triples_np[b_idx]
                # Check if any triple in this block is exclusively covered by it
                has_exclusive = np.any(cover_count[trs] == 1)
                
                if not has_exclusive:
                    cover_count[trs] -= 1
                    changed = True
                else:
                    new_indices_tmp.append(b_idx)
                    new_cover_count_tmp[trs] += 1
            
            if changed:
                new_indices = new_indices_tmp
                # Recompute cover_count from scratch for next iteration
                cover_count = np.zeros(num_triples, dtype=np.int32)
                for b_idx in new_indices:
                    cover_count[block_triples_np[b_idx]] += 1
                new_cover_count = new_cover_count_tmp
            else:
                break
        
        return new_indices, new_cover_count
>>>>>>> REPLACE

