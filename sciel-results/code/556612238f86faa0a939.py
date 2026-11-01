    def optimize_blocks(blocks):
        """Remove redundant blocks efficiently."""
        if not blocks:
            return blocks
        
        # Map block index to set of t-subset indices it covers
        block_tsets = []
        for b in blocks:
            tsets = set()
            for ts in itertools.combinations(b, t):
                tsets.add(t_subset_index[ts])
            block_tsets.append(tsets)
            
        # Count coverage
        cover_count = [0] * total_t
        for tsets in block_tsets:
            for idx in tsets:
                cover_count[idx] += 1
                
        kept_indices = []
        for i in range(len(blocks)):
            if time.time() - start_time > DEADLINE:
                break
            redundant = True
            for idx in block_tsets[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if not redundant:
                kept_indices.append(i)
            else:
                # Decrement cover counts for this block
                for idx in block_tsets[i]:
                    cover_count[idx] -= 1
                    
        return [blocks[i] for i in kept_indices]
