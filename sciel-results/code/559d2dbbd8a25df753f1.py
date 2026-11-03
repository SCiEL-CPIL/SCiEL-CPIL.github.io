<<<<<<< SEARCH
    def local_search(blocks_idx, seed):
        random.seed(seed)
        blocks_idx = list(blocks_idx)
        
        # Initialize counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in block_to_triples[b_idx]:
                counts[t_idx] += 1
        
        improved = True
        max_iterations = 1000 # Safety
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try to remove or swap blocks
            # Iterate in random order
            random.shuffle(blocks_idx)
            
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                
                # Find essential triples for this block
                # Essential if counts[t] == 1
                trs = block_to_triples[b_idx]
                essential_trs = [t_idx for t_idx in trs if counts[t_idx] == 1]
                
                if not essential_trs:
                    # Can remove
                    for t_idx in trs:
                        counts[t_idx] -= 1
                    blocks_idx.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Try to find a replacement block that covers all essential_trs
                    # Intersection of triple_to_blocks for each essential triple
                    # Start with blocks covering the first essential triple
                    candidates = set(triple_to_blocks[essential_trs[0]])
                    
                    for t_idx in essential_trs[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Pick a random candidate
                        # Prefer one that is not already in blocks_idx?
                        # If it is in blocks_idx, we can just remove b_idx?
                        # No, if b_new is in blocks_idx, then essential_trs would have count >= 2.
                        # So b_new must be new.
                        b_new = random.choice(list(candidates))
                        
                        # Replace b_idx with b_new
                        # Remove b_idx
                        for t_idx in trs:
                            counts[t_idx] -= 1
                        # Add b_new
                        for t_idx in block_to_triples[b_new]:
                            counts[t_idx] += 1
                        
                        blocks_idx[i] = b_new
                        improved = True
                        break # Restart loop
                        
        return blocks_idx
=======
    def local_search(blocks_idx, seed):
        random.seed(seed)
        blocks_idx = list(blocks_idx)
        
        # Initialize counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_idx:
            counts[block_to_triples_np[b_idx]] += 1
        
        improved = True
        max_iterations = 2000 # Safety
        
        while improved and max_iterations > 0 and time.time() < deadline:
            improved = False
            max_iterations -= 1
            
            # Try to remove or swap blocks
            # Iterate in random order
            random.shuffle(blocks_idx)
            
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                
                # Find essential triples for this block
                # Essential if counts[t] == 1
                trs = block_to_triples_np[b_idx]
                essential_mask = counts[trs] == 1
                essential_trs = trs[essential_mask]
                
                if len(essential_trs) == 0:
                    # Can remove
                    counts[trs] -= 1
                    blocks_idx.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Try to find a replacement block that covers all essential_trs
                    # Intersection of triple_to_blocks for each essential triple
                    candidates = set(triple_to_blocks[essential_trs[0]])
                    
                    for t_idx in essential_trs[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Pick a random candidate
                        b_new = random.choice(list(candidates))
                        
                        # Replace b_idx with b_new
                        counts[trs] -= 1
                        counts[block_to_triples_np[b_new]] += 1
                        
                        blocks_idx[i] = b_new
                        improved = True
                        break # Restart loop
                        
        return blocks_idx
>>>>>>> REPLACE

