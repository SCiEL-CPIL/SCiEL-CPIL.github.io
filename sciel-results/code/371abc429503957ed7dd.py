<<<<<<< SEARCH
    def minimize(blocks_indices, random_seed=None):
        """
        Remove redundant blocks.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                # If any subset in this block is covered ONLY by this block, it's not redundant
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if redundant_blocks:
                # Remove a random redundant block to break symmetry / explore
                remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                
                cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks
=======
    def minimize(blocks_indices, random_seed=None, deadline=None):
        """
        Remove redundant blocks.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        changed = True
        while changed:
            if deadline is not None and time.time() > deadline:
                break
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if redundant_blocks:
                remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                
                cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks
>>>>>>> REPLACE

