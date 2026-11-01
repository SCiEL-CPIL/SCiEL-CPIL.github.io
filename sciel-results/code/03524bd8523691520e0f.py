import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # block_mask[b] is an integer representing the set of t-subsets covered by block b
    block_mask = [0] * num_k
    # covering_blocks[i] is a list of block indices that cover t-subset i
    covering_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            covering_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        
    def get_unique_mask(counts):
        um = 0
        for i in range(num_t):
            if counts[i] == 1:
                um |= (1 << i)
        return um

    def minimize(active_set, counts, unique_mask=None):
        if unique_mask is None:
            unique_mask = get_unique_mask(counts)
        
        changed = True
        while changed:
            changed = False
            # Iterate over a copy because we modify the set
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                # Check if b_idx has any unique coverage
                if block_mask[b_idx] & unique_mask:
                    continue
                # It has no unique coverage, so it can be removed
                active_set.remove(b_idx)
                bm = block_mask[b_idx]
                # Update counts and unique_mask
                for i in range(num_t):
                    if (bm >> i) & 1:
                        old_count = counts[i]
                        counts[i] = old_count - 1
                        if old_count == 2:
                            # Becomes unique
                            unique_mask |= (1 << i)
                        # If old_count > 2, no change to unique_mask
                        # If old_count == 1, this block had unique coverage, so we wouldn't be here.
                changed = True
        return active_set, counts, unique_mask

    def greedy_construct(noise=0.0):
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        
        while uncovered_mask:
            best_block = -1
            second_best_block = -1
            max_gain = 0
            second_max_gain = 0
            
            # Optimization: only consider blocks that cover at least one uncovered subset
            # We can iterate all blocks, 50k is small
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    second_max_gain = max_gain
                    second_best_block = best_block
                    max_gain = gain
                    best_block = b_idx
                elif gain > second_max_gain:
                    second_max_gain = gain
                    second_best_block = b_idx
            
            if best_block == -1:
                break
                
            # Noisy selection
            chosen_block = best_block
            if noise > 0 and random.random() < noise and second_best_block != -1:
                chosen_block = second_best_block
            
            active_set.add(chosen_block)
            bm = block_mask[chosen_block]
            for i in range(num_t):
                if (bm >> i) & 1:
                    counts[i] += 1
            
            uncovered_mask &= ~bm
            
        return active_set, counts

    def local_search(active_set, counts, unique_mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                if not (block_mask[b_idx] & unique_mask):
                    # Remove
                    active_set.remove(b_idx)
                    bm = block_mask[b_idx]
                    for i in range(num_t):
                        if (bm >> i) & 1:
                            old_count = counts[i]
                            counts[i] = old_count - 1
                            if old_count == 2:
                                unique_mask |= (1 << i)
                    improved = True
                    # After removal, we should probably restart the local search pass
                    # or at least re-minimize
                    active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                # Find the rarest subset in u_b
                # Iterate bits in u_b
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                # Sort by frequency (length of covering_blocks)
                bits.sort(key=lambda i: len(covering_blocks[i]))
                rarest_idx = bits[0]
                
                found_swap = False
                # Try to find a candidate that covers all unique subsets
                for cand_idx in covering_blocks[rarest_idx]:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                        
                    # Check if cand covers all of u_b
                    if (block_mask[cand_idx] & u_b) == u_b:
                        # Perform swap
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        # Update counts
                        bm_out = block_mask[b_idx]
                        bm_in = block_mask[cand_idx]
                        
                        # First, remove b_idx effects
                        for i in range(num_t):
                            if (bm_out >> i) & 1:
                                old_count = counts[i]
                                counts[i] = old_count - 1
                                if old_count == 2:
                                    unique_mask |= (1 << i)
                                elif old_count == 1:
                                    unique_mask &= ~(1 << i)
                        
                        # Then, add cand_idx effects
                        for i in range(num_t):
                            if (bm_in >> i) & 1:
                                old_count = counts[i]
                                counts[i] = old_count + 1
                                if old_count == 0:
                                    unique_mask |= (1 << i)
                                elif old_count == 1:
                                    unique_mask &= ~(1 << i)
                        
                        # Minimize after swap
                        active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
                        improved = True
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try add-minimize: add a block that covers many uncovered subsets, then minimize
                    # This can escape local minima where no single swap works
                    if time.time() < deadline:
                        # Find the block with maximum gain on currently uncovered subsets
                        # But we need to be careful - this might not help if we're already covering everything
                        # Instead, let's try a different approach: find a block that, when added,
                        # allows removal of multiple existing blocks
                        
                        # For now, just try adding the best covering block and minimizing
                        # This is a heuristic that may or may not help
                        pass

        return active_set, counts, unique_mask

    # Main Logic
    best_solution = None
    best_count = float('inf')
    
    # Run multiple restarts with adaptive budget
    # Start with fewer restarts, increase if time permits
    base_restarts = 3
    extra_restarts = 0
    
    for restart_num in range(base_restarts + 10):  # Cap at 13 total
        if time.time() > deadline - 15:
            break
        
        # Use more time for later restarts if we're making progress
        # Introduce noise to escape local minima
        noise = 0.0 if restart_num == 0 else 0.1
        
        active_set, counts = greedy_construct(noise)
        active_set, counts, unique_mask = minimize(active_set, counts)
        active_set, counts, unique_mask = local_search(active_set, counts, unique_mask)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
    
    # Final result
    if best_solution is None:
        # Fallback
        active_set, counts = greedy_construct(0.0)
        active_set, counts, unique_mask = minimize(active_set, counts)
        best_solution = active_set
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result
