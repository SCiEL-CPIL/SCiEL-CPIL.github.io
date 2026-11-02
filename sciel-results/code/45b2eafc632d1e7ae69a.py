import itertools
import random
import time

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    # v=19, k=8, t=5
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each 8-subset, the list of 5-subset indices it covers
    # We don't precompute all 8-subsets, we generate them on the fly or cache.
    # For greedy, we need to find 8-subsets containing a specific 5-subset.
    
    def get_covered_5s(block):
        """Returns list of indices of 5-subsets covered by this 8-block."""
        return [t_subset_to_idx[comb] for comb in itertools.combinations(block, t)]

    # 1. Greedy Construction
    def greedy_construct():
        uncovered = [1] * num_t_subsets
        uncovered_count = num_t_subsets
        blocks = []
        
        # Cache for 8-subsets containing a given 5-subset
        # Actually, just generate them each time.
        
        while uncovered_count > 0:
            # Pick a random uncovered 5-subset
            # To be efficient, keep a list of uncovered indices
            # But updating a list is O(N). 
            # Let's just pick a random index and check if uncovered.
            # With 11628 items, this is fine.
            
            target_idx = -1
            while target_idx == -1:
                cand = random.randint(0, num_t_subsets - 1)
                if uncovered[cand]:
                    target_idx = cand
            
            target_subset = all_t_subsets[target_idx]
            
            # Find 8-subsets containing target_subset
            # Remaining points: v - t
            # Need to choose k - t = 3 more points
            remaining_points = [p for p in points if p not in target_subset]
            candidates = []
            
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_subset + extra))
                # Calculate gain: how many uncovered 5-subsets does this block cover?
                gain = 0
                covered = get_covered_5s(block)
                for idx in covered:
                    if uncovered[idx]:
                        gain += 1
                candidates.append((gain, block))
            
            # Pick the best candidate (max gain)
            # To add randomness, pick from top 10 or so
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_block = candidates[0]
            
            # Add block
            blocks.append(best_block)
            
            # Update uncovered
            covered = get_covered_5s(best_block)
            for idx in covered:
                if uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1
            
        return blocks

    # 2. Minimalization
    def minimalize(blocks, counts, uncovered_count):
        """
        Tries to remove blocks.
        Returns new blocks, counts, uncovered_count.
        """
        # counts: list of size num_t_subsets, counts[i] = number of blocks covering i
        # We need to update counts as we remove.
        
        # To allow different minimalizations, we shuffle the order of checks
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        new_blocks = []
        removed_any = False
        
        # We process blocks one by one.
        # If we remove a block, we must update counts.
        # If we keep it, we do nothing.
        
        # It's safer to build a new list.
        # But counts must reflect the *current* set of blocks being considered.
        # This is tricky if we remove block A, then block B's redundancy might change.
        # Standard approach: Iterate. If block i can be removed (given current counts), remove it.
        
        # Let's use a flag to indicate if we should restart the pass.
        improved = True
        while improved:
            improved = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            # We need to simulate removals.
            # If we remove a block, counts change.
            # Let's make a copy of counts to test removals?
            # Or just test, and if successful, apply.
            
            for i in indices:
                # Check if block i is redundant
                # A block is redundant if for all s in block_i, counts[s] > 1
                # OR if removing it doesn't make any count 0.
                
                block = blocks[i]
                covered = get_covered_5s(block)
                
                is_redundant = True
                for idx in covered:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    # Update counts
                    for idx in covered:
                        counts[idx] -= 1
                        # uncovered_count logic: 
                        # If counts[idx] becomes 0, uncovered_count increases.
                        # But we checked is_redundant, so counts[idx] was > 1, so now >= 1.
                        # So uncovered_count does NOT change.
                    
                    # Remove from blocks list
                    # Since we are iterating over indices, we need to be careful.
                    # Let's collect blocks to keep.
                    # Actually, modifying list while iterating is bad.
                    # Let's use a boolean mask or build new list.
                    
                    # For simplicity in this loop, we just mark it for removal
                    # and update counts.
                    # We'll filter at the end of the pass.
                    blocks[i] = None 
                    improved = True
                    # Note: counts are updated. Subsequent checks in this pass use updated counts.
                    # This is valid "greedy deletion".
            
            # Filter out None blocks
            if improved:
                blocks = [b for b in blocks if b is not None]
        
        return blocks, counts, uncovered_count

    # 3. Local Search with Swaps
    def local_search(blocks, time_limit):
        # Initialize counts
        counts = [0] * num_t_subsets
        for block in blocks:
            covered = get_covered_5s(block)
            for idx in covered:
                counts[idx] += 1
        
        uncovered_count = 0
        for c in counts:
            if c == 0:
                uncovered_count += 1
        
        # Ensure we start valid
        assert uncovered_count == 0, "Greedy construction failed to cover all subsets"
        
        best_blocks = list(blocks)
        best_len = len(blocks)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit - 10: # 10s margin
            # 1. Minimalize
            blocks, counts, uncovered_count = minimalize(blocks, counts, uncovered_count)
            
            if len(blocks) < best_len:
                best_blocks = list(blocks)
                best_len = len(blocks)
                # print(f"Improved: {best_len}")
            
            # 2. Try Swaps
            # Pick a random block to modify
            if len(blocks) > 0:
                idx = random.randint(0, len(blocks) - 1)
                block = blocks[idx]
                
                # Try a few random swaps
                swaps_tried = 0
                swap_applied = False
                max_swaps = 50
                while swaps_tried < max_swaps and not swap_applied:
                    swaps_tried += 1
                    # Pick x in block, y not in block
                    x = random.choice(block)
                    others = [p for p in points if p not in block]
                    if not others: continue
                    y = random.choice(others)
                    
                    new_block = tuple(sorted((block - {x}) | {y}))
                    
                    # Check if new_block is same as old
                    if set(new_block) == set(block): continue
                    
                    # Check validity of swap
                    # Old block covers: old_cov
                    # New block covers: new_cov
                    # Condition: For all s in old_cov \ new_cov, counts[s] > 1
                    # (If counts[s] == 1, it means only old_block covers s. If we remove old_block, s is lost unless new_block covers it.)
                    # So: if s in old_cov and s not in new_cov, we need counts[s] > 1.
                    
                    old_cov = set(get_covered_5s(block))
                    new_cov = set(get_covered_5s(new_block))
                    
                    valid = True
                    for s in old_cov:
                        if s not in new_cov:
                            if counts[s] == 1:
                                valid = False
                                break
                    
                    if valid:
                        # Apply swap
                        # Remove old block
                        for s in old_cov:
                            counts[s] -= 1
                        # Add new block
                        for s in new_cov:
                            counts[s] += 1
                        
                        blocks[idx] = new_block
                        swap_applied = True
                        break
            
            # If we applied a swap, we might have created redundancies, so minimalize in next loop.
            # If we didn't apply a swap, we are stuck in a local minimum for swaps.
            # We can try a more aggressive perturbation: 
            # Randomly change 2 points? Or restart from best?
            # Let's just continue. The shuffle in minimalize helps.
            
            # If we are stuck (no swap applied and no minimalization improvement), 
            # we might want to randomize the current state slightly to escape.
            # For now, just loop.
            
        return best_blocks

    # Execute
    initial_blocks = greedy_construct()
    result_blocks = local_search(initial_blocks, 140) # Leave 10s for safety
    
    # Final minimalization pass
    counts = [0] * num_t_subsets
    for block in result_blocks:
        covered = get_covered_5s(block)
        for idx in covered:
            counts[idx] += 1
    
    result_blocks, _, _ = minimalize(result_blocks, counts, 0)
    
    # Return as lists of ints
    return [list(b) for b in result_blocks]

# EVOLVE-BLOCK-END
