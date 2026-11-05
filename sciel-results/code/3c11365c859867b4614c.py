import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block which t-subsets it covers
    # Store as list of lists of t-subset indices
    block_ts_indices = []
    for block in k_subsets:
        ts_indices = []
        for t_sub in itertools.combinations(block, t):
            ts_indices.append(t_subset_to_idx[t_sub])
        block_ts_indices.append(ts_indices)
    
    # Also precompute bitmasks for fast intersection
    block_masks = []
    for ts_indices in block_ts_indices:
        mask = 0
        for idx in ts_indices:
            mask |= (1 << idx)
        block_masks.append(mask)
    
    full_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Strategy: Randomized Greedy with Local Search
    # We will run multiple iterations. In each iteration:
    # 1. Perform a greedy set cover construction, but with randomization in tie-breaking.
    # 2. Perform a local search to remove redundant blocks.
    # 3. Try to swap blocks to improve coverage or reduce count.
    
    runs = 0
    while time.time() < deadline:
        runs += 1
        uncovered = full_mask
        selected_indices = []
        
        # To randomize, we shuffle the order in which we consider blocks for the "max gain" step.
        # Instead of scanning all blocks every time (which is O(N * K) total),
        # we can just scan all blocks. N=50388, K~150. 50k * 150 = 7.5M ops.
        # In Python, this is slow.
        
        # Optimization:
        # We can maintain a list of "candidate" blocks?
        # Or, we can use a "lazy" deletion approach?
        # Actually, the standard greedy for set cover is:
        # while uncovered:
        #   pick block with max |block & uncovered|
        #   remove those from uncovered
        
        # To speed this up, we can use the fact that we only need the MAX.
        # We can iterate through blocks and keep track of the best.
        # But we must do this for every step.
        
        # Let's try to optimize the inner loop.
        # Instead of bit_count on a large int, can we do better?
        # Maybe pre-compute the "coverage count" for each t-subset?
        # No, that's for the dual greedy.
        
        # Let's stick to the bitmask greedy but optimize the loop.
        # We can skip blocks that are already selected?
        # We can also stop early if we find a block that covers ALL remaining uncovered? (Unlikely)
        
        # Let's implement the randomized greedy.
        # Randomization: Shuffle the block indices at the start of the run.
        # This ensures that in case of ties, different blocks are picked.
        
        indices = list(range(num_blocks))
        random.shuffle(indices)
        
        while uncovered and time.time() < deadline:
            best_gain = -1
            best_block_idx = -1
            
            # Iterate over shuffled indices
            # We can break early if we find a block that covers all uncovered?
            # If gain == uncovered.bit_count(), we can break.
            
            for i in indices:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # Early termination if we cover everything?
                    if gain == uncovered.bit_count():
                        break
            
            if best_gain == 0:
                break
            
            selected_indices.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
            # Remove selected block from indices?
            # This would allow us to reuse the "shuffle" logic more efficiently?
            # No, we just need to ensure we don't pick the same block again.
            # Since we update `uncovered`, the gain for a previously selected block
            # might be 0 or small, but not necessarily 0 if it overlaps with new uncovered parts?
            # Wait, `uncovered` only shrinks. So if a block was selected before,
            # its intersection with the NEW uncovered set is a subset of its intersection with the OLD uncovered set.
            # It's possible it still has positive gain if it wasn't "fully consumed"?
            # No, we subtract its mask from uncovered. So any t-subset it covered is no longer uncovered.
            # So its gain against the NEW uncovered set is 0?
            # Yes! Because `uncovered` removes exactly the bits set in `block_masks[best_block_idx]`.
            # So `(uncovered & block_masks[best_block_idx])` will be 0.
            # Therefore, we don't need to explicitly remove it, it will naturally have gain 0.
            # But we still iterate over it.
            # To save time, we could maintain a list of active blocks.
            # But removing from a list is O(N).
            # Given N=50k, and 150 steps, 50k*150 = 7.5M iterations.
            # Each iteration does a bit_count.
            # In Python, this might take ~ 1-2 seconds per run.
            # We have 140 seconds. So we can do ~ 50-100 runs.
            
        # Post-processing: Remove redundant blocks
        if time.time() < deadline:
            # Rebuild counts
            counts = [0] * num_t_subsets
            block_ts_list = []
            for idx in selected_indices:
                ts_list = block_ts_indices[idx]
                block_ts_list.append(ts_list)
                for ts_idx in ts_list:
                    counts[ts_idx] += 1
            
            active = [True] * len(selected_indices)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(selected_indices)):
                    if not active[i]:
                        continue
                    redundant = True
                    for ts_idx in block_ts_list[i]:
                        if counts[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        active[i] = False
                        changed = True
                        for ts_idx in block_ts_list[i]:
                            counts[ts_idx] -= 1
            
            final_indices = [idx for i, idx in enumerate(selected_indices) if active[i]]
            
            # Local Search: Try to swap blocks
            # Idea: For each block in the solution, try to replace it with a different block
            # that covers the unique t-subsets of the removed block,
            # and see if it allows removing another block?
            # This is complex.
            
            # Simpler: If we have time, try to swap one block for another
            # that covers the "critical" t-subsets of the first,
            # and see if it allows removing another block.
            
            # For now, just update best.
            if len(final_indices) < best_count:
                best_count = len(final_indices)
                best_blocks = [k_subsets[i] for i in final_indices]
                
            # If we found a very good solution, stop early?
            if best_count <= 131: # Lower bound
                break
        
        # To get diversity in the next run, we rely on the shuffle.
        # But shuffling 50k elements is cheap.
        
        # We can also try different random seeds or strategies.
        # For now, just loop.

    if best_blocks is None:
        return []
        
    return best_blocks
