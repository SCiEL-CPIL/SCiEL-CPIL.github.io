import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    # 1. Pre-computation
    # Map t-subsets to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Full mask of all t-subsets
    full_mask = (1 << n_t_subs) - 1
    
    # Map k-blocks to their t-subset indices and bitmask
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_data = {} # block_tuple -> (list_of_t_sub_indices, bitmask)
    
    # Also map t-subset -> list of blocks containing it (for greedy heuristic)
    t_sub_to_blocks = {i: [] for i in range(n_t_subs)}
    
    for b in all_k_blocks:
        subs = list(itertools.combinations(b, t))
        indices = [t_sub_to_idx[s] for s in subs]
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(b)
        block_data[b] = (indices, mask)
        
    # 2. Greedy Construction (with randomization for diversity)
    def greedy_construct(randomize=False):
        uncovered_mask = full_mask
        blocks = []
        used_blocks = set()
        
        while uncovered_mask:
            if randomize and random.random() < 0.5:
                # Pick a random uncovered subset
                # Find an index set in uncovered_mask
                # This is slow if done naively. 
                # Alternative: Pick a random block from the pool? No.
                # Let's stick to deterministic greedy but maybe randomize tie-breaking?
                # Or just run multiple greedy passes with different random seeds for local search?
                pass
            
            # Heuristic: Pick the uncovered subset with the fewest available blocks?
            # Or just pick the first uncovered subset?
            # Let's find the first uncovered subset index
            # We can find the lowest set bit
            # lowest_bit = uncovered_mask & -uncovered_mask
            # idx = lowest_bit.bit_length() - 1
            
            # To make it faster, we can just pick a random uncovered subset?
            # Getting a random bit from a large int is tricky.
            # Let's just pick the lowest set bit for determinism in the base greedy,
            # and rely on local search for optimization.
            
            # Optimization: Instead of finding the "best" block globally,
            # pick a random uncovered subset, then pick the best block covering it.
            # This is O(C(v-t, k-t)) per step.
            
            # Get a random uncovered subset index
            # We can iterate through set bits?
            # Simpler: Just pick the first one.
            # lowest_bit = uncovered_mask & -uncovered_mask
            # idx = lowest_bit.bit_length() - 1
            
            # Actually, standard greedy (max gain) is better.
            # But iterating all 496k blocks to find max gain is slow.
            # Let's use the "pick random uncovered subset, then best block covering it" strategy.
            
            # To pick a random uncovered subset efficiently:
            # We can maintain a list of uncovered indices?
            # Or just pick the lowest set bit. It's deterministic but works.
            # Let's try to pick a random one by sampling?
            # For now, let's use the lowest set bit for speed and simplicity in the greedy phase,
            # as the local search is the main driver of quality.
            
            lowest_bit = uncovered_mask & -uncovered_mask
            idx = lowest_bit.bit_length() - 1
            
            candidates = t_sub_to_blocks[idx]
            best_block = None
            best_gain = -1
            
            for b in candidates:
                if b in used_blocks:
                    continue
                _, mask = block_data[b]
                gain = (mask & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block is None:
                break # Should not happen
                
            blocks.append(best_block)
            used_blocks.add(best_block)
            
            _, mask = block_data[best_block]
            uncovered_mask &= ~mask
            
        return blocks

    # 3. Local Search
    def local_search(blocks):
        if not blocks:
            return blocks
            
        # Initialize counts
        counts = [0] * n_t_subs
        for b in blocks:
            indices, _ = block_data[b]
            for idx in indices:
                counts[idx] += 1
                
        def is_redundant(b):
            indices, _ = block_data[b]
            for idx in indices:
                if counts[idx] <= 1:
                    return False
            return True
            
        improved = True
        while improved:
            improved = False
            # Try to remove any block
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i]):
                    b = blocks[i]
                    indices, _ = block_data[b]
                    for idx in indices:
                        counts[idx] -= 1
                    del blocks[i]
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Try swaps
            # Random restarts within local search?
            # Let's try a few random swaps
            
            for _ in range(100): # Limit swaps per pass to avoid infinite loops on bad moves?
                if time.time() - start_time > time_limit:
                    return blocks
                    
                if not blocks:
                    break
                    
                block_idx = random.randint(0, len(blocks) - 1)
                block = blocks[block_idx]
                x_idx = random.randint(0, k - 1)
                x = block[x_idx]
                
                candidates = [y for y in range(v) if y not in block]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                new_block_list = [p for p in block if p != x]
                new_block_list.append(y)
                new_block = tuple(sorted(new_block_list))
                
                # Check if swap is valid (doesn't uncover any subset)
                # Subsets in old block containing x must have count > 1
                # Subsets in new block must be covered by others OR be in old block (if not containing x? No, new block is different)
                
                # Actually, we need to check:
                # 1. All subsets in new_block are covered by (blocks \ {block}) U {new_block}
                # 2. All subsets in old_block are covered by (blocks \ {block}) U {new_block}
                
                # Simpler check:
                # Remove block from counts
                old_indices, _ = block_data[block]
                for idx in old_indices:
                    counts[idx] -= 1
                    
                # Check if any subset in old_block is now uncovered (count == 0)
                valid = True
                for idx in old_indices:
                    if counts[idx] == 0:
                        valid = False
                        break
                        
                if not valid:
                    # Revert
                    for idx in old_indices:
                        counts[idx] += 1
                    continue
                    
                # Add new_block to counts
                new_indices, _ = block_data[new_block]
                for idx in new_indices:
                    counts[idx] += 1
                    
                # Now check if we can remove any block (including the new one? No, we just added it)
                # Or if the new block is redundant?
                # Or if any *other* block is now redundant?
                
                # Let's just check if new_block is redundant
                if is_redundant(new_block):
                    # Remove it
                    for idx in new_indices:
                        counts[idx] -= 1
                    # We effectively replaced block with nothing? No, we replaced block with new_block, then removed new_block.
                    # So we removed the original block.
                    # This is a valid deletion!
                    del blocks[block_idx]
                    improved = True
                    break # Restart the while loop
                    
                # Check if any other block is redundant
                found_redundant = -1
                for i in range(len(blocks)):
                    if i == block_idx:
                        continue
                    if is_redundant(blocks[i]):
                        found_redundant = i
                        break
                        
                if found_redundant != -1:
                    # Accept swap and remove redundant block
                    blocks[block_idx] = list(new_block)
                    # Note: counts currently has new_block added and old_block removed.
                    # We need to update counts to reflect the permanent state.
                    # Currently:
                    # counts = (initial - old) + new
                    # We remove blocks[found_redundant]
                    b_to_remove = blocks[found_redundant]
                    rem_indices, _ = block_data[b_to_remove]
                    for idx in rem_indices:
                        counts[idx] -= 1
                    del blocks[found_redundant]
                    improved = True
                    break
                    
                # If no improvement, revert
                for idx in new_indices:
                    counts[idx] -= 1
                for idx in old_indices:
                    counts[idx] += 1
                
        return blocks

    best_blocks = []
    
    # Run multiple greedy + local search iterations
    iterations = 0
    while time.time() - start_time < time_limit:
        # Greedy
        blocks = greedy_construct()
        
        # Local Search
        blocks = local_search(blocks)
        
        if not best_blocks or len(blocks) < len(best_blocks):
            best_blocks = blocks
            # print(f"New best: {len(best_blocks)}")
            
        iterations += 1
        # If we are converging, maybe break?
        if iterations > 10 and len(best_blocks) == 76: # Known best is 76, if we hit it, we can try to beat it, but 76 is hard.
             # Actually, 76 is the current best known. We aim for < 76.
             pass

    return [list(b) for b in best_blocks]
