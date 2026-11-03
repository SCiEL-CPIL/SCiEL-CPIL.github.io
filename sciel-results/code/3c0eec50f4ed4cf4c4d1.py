<<<<<<< SEARCH
    def local_search_improve(initial_blocks, time_limit):
        """
        Simulated Annealing local search.
        Moves: Remove one block, Add one block.
        Optimized revert logic to avoid full re-computation of uncovered count.
        """
        if not initial_blocks:
            return initial_blocks

        current_blocks = list(initial_blocks)
        current_counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_blocks:
            current_counts[block_triples[b_idx]] += 1
            
        current_uncovered = int(np.sum(current_counts == 0))
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        
        M = 1000  # Penalty for uncovered triples
        current_obj = current_size + M * current_uncovered
        best_obj = current_obj
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        all_block_indices = list(range(n_blocks))
        
        start_ls = time.time()
        
        while time.time() < start_ls + time_limit:
            if T < min_T:
                # Restart temperature if stuck? Or just break.
                # Resetting T allows escaping deep local minima, but risks oscillation.
                # Let's just break or restart gently.
                T = 0.5
                
            if not current_blocks:
                break
                
            # Pick a random block to remove
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            # Pick a random block to add
            add_block = random.choice(all_block_indices)
            
            if add_block == remove_block:
                continue
                
            old_obj = current_obj
            old_uncovered = current_uncovered
            old_size = current_size
            
            # --- Remove ---
            rem_triples = block_triples[remove_block]
            delta_uncovered_remove = 0
            for t_idx in rem_triples:
                if current_counts[t_idx] == 1:
                    delta_uncovered_remove += 1
                current_counts[t_idx] -= 1
                
            current_blocks.pop(remove_idx_pos)
            current_size -= 1
            current_uncovered += delta_uncovered_remove
            
            # --- Add ---
            add_triples = block_triples[add_block]
            delta_uncovered_add = 0
            for t_idx in add_triples:
                if current_counts[t_idx] == 0:
                    delta_uncovered_add += 1
                current_counts[t_idx] += 1
                
            current_blocks.append(add_block)
            current_size += 1
            current_uncovered -= delta_uncovered_add
            
            # --- Objective ---
            new_obj = current_size + M * current_uncovered
            
            # Update best
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_blocks = list(current_blocks)
                best_obj = new_obj
                
            # Acceptance
            delta = new_obj - old_obj
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    p = math.exp(-delta / T)
                    accept = random.random() < p
                else:
                    accept = False
            
            if not accept:
                # Revert
                # Undo Add
                for t_idx in add_triples:
                    current_counts[t_idx] -= 1
                current_blocks.pop()
                
                # Undo Remove
                for t_idx in rem_triples:
                    current_counts[t_idx] += 1
                current_blocks.insert(remove_idx_pos, remove_block)
                
                current_size = old_size
                current_uncovered = old_uncovered
                current_obj = old_obj
            else:
                current_obj = new_obj
                
            T *= cooling_rate

        return best_blocks
=======
    def local_search_improve(initial_blocks, time_limit):
        """
        Simulated Annealing local search with smart move selection.
        Moves: Remove one block, Add one block (prefer high-overlap additions).
        Also tries pure removals when possible.
        """
        if not initial_blocks:
            return initial_blocks

        current_blocks = list(initial_blocks)
        current_counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_blocks:
            current_counts[block_triples[b_idx]] += 1
            
        current_uncovered = int(np.sum(current_counts == 0))
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        
        M = 10000  # Penalty for uncovered triples (higher to stay valid longer)
        current_obj = current_size + M * current_uncovered
        best_obj = current_obj
        
        T = 2.0
        cooling_rate = 0.9997
        min_T = 0.01
        
        all_block_indices = list(range(n_blocks))
        
        start_ls = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < start_ls + time_limit:
            if T < min_T:
                T = 1.0
                no_improve_count = 0
                
            if not current_blocks:
                break
                
            # Choose move type: 70% swap (remove+add), 30% try removal
            move_type = random.random()
            
            if move_type < 0.3:
                # Try pure removal: pick a block whose all triples have count > 1
                removable_candidates = []
                for pos, b_idx in enumerate(current_blocks):
                    bt = block_triples[b_idx]
                    if all(current_counts[ti] > 1 for ti in bt):
                        removable_candidates.append(pos)
                
                if removable_candidates:
                    remove_idx_pos = random.choice(removable_candidates)
                    remove_block = current_blocks[remove_idx_pos]
                    add_block = None
                else:
                    # Fall back to swap
                    move_type = 0.5
            
            if move_type >= 0.3:
                # Swap: remove one, add one
                remove_idx_pos = random.randint(0, len(current_blocks) - 1)
                remove_block = current_blocks[remove_idx_pos]
                
                # Prefer adding blocks that overlap significantly with current set
                # to increase chance of net size reduction
                if random.random() < 0.6 and current_blocks:
                    # Pick a block that shares many triples with a random current block
                    ref_block = random.choice(current_blocks)
                    ref_triples_set = set(block_triples[ref_block])
                    # Sample a few candidates and pick best overlap
                    candidates = random.sample(all_block_indices, min(20, n_blocks))
                    best_add = None
                    best_overlap = -1
                    for cand in candidates:
                        if cand == remove_block:
                            continue
                        cand_triples = block_triples[cand]
                        overlap = sum(1 for ti in cand_triples if ti in ref_triples_set)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_add = cand
                    if best_add is not None:
                        add_block = best_add
                    else:
                        add_block = random.choice(all_block_indices)
                else:
                    add_block = random.choice(all_block_indices)
                
                if add_block == remove_block:
                    continue
            
            old_obj = current_obj
            old_uncovered = current_uncovered
            old_size = current_size
            
            # --- Remove ---
            rem_triples = block_triples[remove_block]
            delta_uncovered_remove = 0
            for t_idx in rem_triples:
                if current_counts[t_idx] == 1:
                    delta_uncovered_remove += 1
                current_counts[t_idx] -= 1
                
            current_blocks.pop(remove_idx_pos)
            current_size -= 1
            current_uncovered += delta_uncovered_remove
            
            # --- Add (if applicable) ---
            if add_block is not None:
                add_triples = block_triples[add_block]
                delta_uncovered_add = 0
                for t_idx in add_triples:
                    if current_counts[t_idx] == 0:
                        delta_uncovered_add += 1
                    current_counts[t_idx] += 1
                
                current_blocks.append(add_block)
                current_size += 1
                current_uncovered -= delta_uncovered_add
            
            # --- Objective ---
            new_obj = current_size + M * current_uncovered
            
            # Update best
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_blocks = list(current_blocks)
                best_obj = new_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Acceptance
            delta = new_obj - old_obj
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    p = math.exp(-delta / T)
                    accept = random.random() < p
                else:
                    accept = False
            
            if not accept:
                # Revert
                if add_block is not None:
                    # Undo Add
                    for t_idx in add_triples:
                        current_counts[t_idx] -= 1
                    current_blocks.pop()
                
                # Undo Remove
                for t_idx in rem_triples:
                    current_counts[t_idx] += 1
                current_blocks.insert(remove_idx_pos, remove_block)
                
                current_size = old_size
                current_uncovered = old_uncovered
                current_obj = old_obj
            else:
                current_obj = new_obj
                
            T *= cooling_rate

        # Final exact removal pass
        best_blocks = exact_remove_pass(best_blocks)
        
        return best_blocks
>>>>>>> REPLACE

