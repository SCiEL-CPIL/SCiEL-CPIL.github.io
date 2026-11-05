import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each k-subset, the list of t-subsets it covers
    # This might be memory intensive if v,k are large, but v=19, k=7 is small.
    # C(19,7) = 50388 blocks. Each has C(7,4)=35 t-subsets.
    # Total entries ~ 1.7M. This is fine.
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_to_t_subsets = {}
    for block in all_k_subsets:
        t_subs = list(itertools.combinations(block, t))
        block_to_t_subsets[block] = [t_subset_to_idx[s] for s in t_subs]
        
    # Map from t-subset index to list of blocks that contain it
    # This helps in checking redundancy quickly? Maybe not directly.
    # For greedy, we need to know which blocks cover which uncovered t-subsets.
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy iterations with randomization
    for i in range(500): # Max iterations, will break on time
        if time.time() > deadline:
            break
            
        # Randomize the order of t-subsets to break ties and explore different paths
        # Or randomize the order of blocks considered.
        # Let's randomize the initial "uncovered" set order? No, it's a set.
        # We can add a small random noise to the "gain" calculation.
        
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # Greedy selection
        while uncovered:
            # Find block with max coverage of uncovered t-subsets
            # To speed up, we can precompute the "potential" of each block?
            # But uncovered changes.
            # For v=19, k=7, t=4, num_blocks=50k.
            # Iterating 50k blocks for each step (max 151 steps) is 7.5M operations.
            # This is fast in Python if optimized, but might be slow.
            # Let's try to optimize the inner loop.
            
            best_block = None
            max_gain = -1
            
            # Optimization: Instead of iterating all blocks, we can iterate over uncovered t-subsets
            # and find blocks that cover them.
            # But a block covers multiple t-subsets.
            
            # Let's stick to iterating blocks for simplicity first, and see if it's fast enough.
            # 50k blocks * 151 steps * 500 restarts = 3.7 Billion operations. Too slow.
            
            # We need a faster greedy.
            # Precompute for each t-subset, the list of blocks that contain it.
            t_subset_to_blocks = {i: [] for i in range(num_t_subsets)}
            for block, t_subs in block_to_t_subsets.items():
                for idx in t_subs:
                    t_subset_to_blocks[idx].append(block)
            
            # Now, for each step:
            # Count how many uncovered t-subsets each block covers.
            # Initialize a counter for all blocks to 0.
            # For each uncovered t-subset, increment the counter for all blocks that contain it.
            # This is efficient if the number of uncovered t-subsets is small.
            # Initially, it's 3876. Each t-subset is in C(19-4, 7-4) = C(15,3) = 455 blocks.
            # So 3876 * 455 ~ 1.7M increments.
            # Then find the max.
            # Then remove covered t-subsets, and decrement counters for blocks that covered them?
            # This is getting complex to maintain.
            
            # Alternative: Just iterate all blocks and compute gain on the fly.
            # Gain = number of t-subsets in block that are in uncovered.
            # If we represent `uncovered` as a set of indices, we can do:
            # gain = sum(1 for idx in block_to_t_subsets[block] if idx in uncovered)
            # This is 35 checks per block. 50k * 35 = 1.75M checks per step.
            # 151 steps * 1.75M = 265M checks per restart.
            # 500 restarts = 132 Billion checks. Too slow.
            
            # We need to reduce the number of restarts or make the greedy faster.
            # Let's try fewer restarts (e.g., 50) and see if it's fast enough.
            # Or, use a heuristic to prune blocks.
            
            # Let's try a different strategy:
            # 1. Greedy without restarts, but with a "lookahead" or "weighting".
            # 2. Or, use the `t_subset_to_blocks` structure to update gains incrementally.
            
            # Let's implement the incremental update.
            # `block_gain` is a dict or array of length num_blocks.
            # Initially, for each block, gain = number of t-subsets in it.
            # When we add a block, we remove its t-subsets from `uncovered`.
            # For each removed t-subset, we decrement the gain of all blocks that contain it.
            
            # This is efficient!
            # 1. Initialize `block_gain` for all blocks.
            # 2. While uncovered:
            #    a. Find block with max gain.
            #    b. Add to blocks.
            #    c. For each t-subset in this block:
            #       - If it was uncovered, mark as covered.
            #       - For each block that contains this t-subset, decrement its gain.
            
            # Let's implement this.
            
            # Precompute block indices for faster access
            block_idx_map = {block: i for i, block in enumerate(all_k_subsets)}
            num_blocks = len(all_k_subsets)
            
            # block_gain[i] = number of uncovered t-subsets covered by block i
            block_gain = [len(block_to_t_subsets[b]) for b in all_k_subsets]
            
            # t_subset_to_blocks is already a list of block tuples. Convert to indices?
            # Or just use the tuples.
            
            while uncovered:
                # Find max gain block
                # To speed up, we can keep a priority queue, but gains change.
                # For now, linear scan. 50k scan is fast.
                max_gain = -1
                best_block_idx = -1
                for idx in range(num_blocks):
                    if block_gain[idx] > max_gain:
                        max_gain = block_gain[idx]
                        best_block_idx = idx
                        
                if max_gain == 0:
                    # Should not happen if covering is possible
                    break
                    
                best_block = all_k_subsets[best_block_idx]
                blocks.append(best_block)
                
                # Update uncovered and gains
                for t_idx in block_to_t_subsets[best_block]:
                    if t_idx in uncovered:
                        uncovered.remove(t_idx)
                        # Decrement gain for all blocks containing this t-subset
                        for b_tuple in t_subset_to_blocks[t_idx]:
                            b_idx = block_idx_map[b_tuple]
                            block_gain[b_idx] -= 1
                            
                # Set gain of the chosen block to 0? No, it might still cover other uncovered subsets?
                # No, we just removed all its t-subsets from uncovered (if they were there).
                # So its gain should be 0 now.
                block_gain[best_block_idx] = 0
                
            # End of greedy
            
            # Local Search: Remove redundant blocks
            # Repeat until no more blocks can be removed
            improved = True
            while improved:
                improved = False
                for i in range(len(blocks)):
                    # Check if block i is redundant
                    # i.e., all its t-subsets are covered by other blocks
                    # We can check this by seeing if for each t-subset in block i,
                    # there is another block in the covering that contains it.
                    
                    # Efficient check:
                    # For each t-subset in block i, count how many blocks in the covering contain it.
                    # If count > 1, it's covered by others.
                    # We can precompute this?
                    
                    # Simple check:
                    redundant = True
                    for t_sub in itertools.combinations(blocks[i], t):
                        # Check if any other block contains t_sub
                        found = False
                        for j in range(len(blocks)):
                            if i == j:
                                continue
                            if t_sub in blocks[j]: # This is slow (tuple in list)
                                found = True
                                break
                        if not found:
                            redundant = False
                            break
                            
                    if redundant:
                        blocks.pop(i)
                        improved = True
                        break # Restart the loop as indices changed
                        
            # End local search
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks.copy()
                
    if best_blocks is None:
        # Fallback: return a valid covering (even if not optimal)
        # The greedy should always find one.
        return []
        
    return best_blocks
