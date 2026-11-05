<<<<<<< SEARCH
    def local_search_swap(blocks_indices, max_swaps, random_seed):
        """
        Perform local search using swaps.
        Returns a new list of blocks if improved, else None.
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
            
        block_set = set(current_blocks)
        
        # Try to find swaps
        # We iterate through blocks and try to replace them
        # To avoid O(N^2), we sample blocks to try
        num_tries = min(len(current_blocks) * 2, 50)
        
        improved = False
        for _ in range(num_tries):
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap
            idx = rng.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[idx]
            b_subs = block_to_subsets_arr[b_idx]
            
            # Find subsets in b_idx that are covered ONLY by b_idx
            critical_subsets = b_subs[cover_count[b_subs] == 1]
            
            if len(critical_subsets) == 0:
                # This block is redundant!
                # Remove it
                cover_count[b_subs] -= 1
                current_blocks.pop(idx)
                block_set.remove(b_idx)
                improved = True
                continue
            
            # Find a replacement block B that covers all critical_subsets
            # Candidate blocks are the intersection of subset_to_blocks[s] for s in critical_subsets
            # Start with the set of blocks covering the first critical subset
            candidate_blocks = set(subset_to_blocks_arr[critical_subsets[0]].tolist())
            
            for s in critical_subsets[1:]:
                candidate_blocks.intersection_update(subset_to_blocks_arr[s].tolist())
                if not candidate_blocks:
                    break
            
            if not candidate_blocks:
                continue
                
            # Remove current blocks from candidates
            candidate_blocks.difference_update(block_set)
            
            if not candidate_blocks:
                continue
                
            # Pick a candidate
            # Prefer one that covers many other subsets (high "weight") to help future deletions
            # Or just random
            candidates_list = list(candidate_blocks)
            new_b_idx = candidates_list[rng.randint(0, len(candidates_list) - 1)]
            
            # Perform swap
            # Remove old
            cover_count[block_to_subsets_arr[b_idx]] -= 1
            current_blocks[idx] = new_b_idx
            block_set.remove(b_idx)
            block_set.add(new_b_idx)
            # Add new
            cover_count[block_to_subsets_arr[new_b_idx]] += 1
            improved = True
            
        return current_blocks
=======
    def local_search_swap(blocks_indices, max_swaps, random_seed):
        """
        Perform local search using swaps.
        Returns a new list of blocks if improved, else None.
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
            b_row = block_to_subsets_sparse.getrow(b_idx)
            cover_count[b_row.indices] += 1
            
        block_set = set(current_blocks)
        
        # Try to find swaps
        num_tries = min(len(current_blocks) * 2, 50)
        
        improved = False
        for _ in range(num_tries):
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap
            idx = rng.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[idx]
            b_row = block_to_subsets_sparse.getrow(b_idx)
            b_subs = b_row.indices
            
            # Find subsets in b_idx that are covered ONLY by b_idx
            critical_subsets = b_subs[cover_count[b_subs] == 1]
            
            if len(critical_subsets) == 0:
                # This block is redundant!
                cover_count[b_subs] -= 1
                current_blocks.pop(idx)
                block_set.remove(b_idx)
                improved = True
                continue
            
            # Find a replacement block B that covers all critical_subsets
            candidate_blocks = set(subset_to_blocks_arr[critical_subsets[0]].tolist())
            
            for s in critical_subsets[1:]:
                candidate_blocks.intersection_update(subset_to_blocks_arr[s].tolist())
                if not candidate_blocks:
                    break
            
            if not candidate_blocks:
                continue
                
            # Remove current blocks from candidates
            candidate_blocks.difference_update(block_set)
            
            if not candidate_blocks:
                continue
                
            # Prefer candidate that covers most uncovered subsets
            candidates_list = list(candidate_blocks)
            best_candidate = candidates_list[0]
            best_gain = -1
            for cand in candidates_list[:20]:  # Limit evaluation
                cand_row = block_to_subsets_sparse.getrow(cand)
                cand_subs = cand_row.indices
                gain = np.sum(cover_count[cand_subs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            new_b_idx = best_candidate
            
            # Perform swap
            cover_count[b_row.indices] -= 1
            current_blocks[idx] = new_b_idx
            block_set.remove(b_idx)
            block_set.add(new_b_idx)
            new_row = block_to_subsets_sparse.getrow(new_b_idx)
            cover_count[new_row.indices] += 1
            improved = True
            
        return current_blocks
>>>>>>> REPLACE

Now let me add the group-based structured construction and improve the main loop:

