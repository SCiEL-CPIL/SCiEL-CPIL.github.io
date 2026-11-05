    # In construct:
    block_id = {block: i for i, block in enumerate(all_blocks)}
    
    def reduce_cover(blocks):
        if not blocks:
            return blocks
        
        triple_counts = np.zeros(n_triples, dtype=np.int32)
        # Store block indices for fast access
        block_indices = [block_id[tuple(b)] for b in blocks]
        
        for bi in block_indices:
            triple_counts[block_triple_indices[bi]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(block_indices)):
                bi = block_indices[i]
                # Check redundancy
                redundant = np.all(triple_counts[block_triple_indices[bi]] > 1)
                
                if redundant:
                    # Remove block i
                    block_indices.pop(i)
                    blocks.pop(i)
                    triple_counts[block_triple_indices[bi]] -= 1
                    changed = True
                    break
                    
        return blocks
