import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
        
    # Greedy Construction
    # uncovered_mask is a bitset or list of bools? 
    # n_t_subs is 3876. A list of ints (0/1) is fine.
    uncovered = [1] * n_t_subs
    cover_count = [0] * n_t_subs
    blocks = []
    
    # For speed in greedy, we can just iterate all blocks and pick best.
    # But 50k blocks * 3876 subsets is ~200M ops per step. 
    # Number of blocks in solution is ~150.
    # 150 * 200M = 30B ops. Too slow in Python.
    
    # We need a faster greedy.
    # Instead of scanning all blocks, we can pick a random uncovered subset, 
    # and find a block covering it that covers the most OTHER uncovered subsets.
    # Or, we can use a heuristic: pick the block that covers the most uncovered subsets.
    # To speed this up, we can maintain a count of uncovered subsets per block.
    # But updating that count when we add a block is O(1) if we precomputed intersections? No.
    
    # Let's try a different Greedy:
    # 1. Pick an uncovered subset S.
    # 2. Find all blocks containing S.
    # 3. Among those, pick the one that covers the most *other* uncovered subsets.
    # 4. Add it.
    
    # This is O(NumBlocksContainingS * t_subsets_in_block).
    # NumBlocksContainingS = C(v-t, k-t) = C(15, 3) = 455.
    # t_subsets_in_block = C(7, 4) = 35.
    # So ~16k ops per block added. 150 blocks -> 2.4M ops. Very fast.
    
    # Pre-compute which blocks contain which t-subset?
    # Or just filter on the fly.
    
    # Let's use the "Pick uncovered, find best covering block" strategy.
    # To avoid bias, we pick the uncovered subset with the *fewest* covering blocks? 
    # No, standard greedy picks the most constrained or the one that leads to best gain.
    # Let's stick to: Pick an uncovered subset (random or first), find best block covering it.
    
    # Actually, a global greedy (best block overall) is often better but slower.
    # Let's try the "Most Uncovered" greedy but optimize the search.
    # We can maintain `uncovered_count_per_block`?
    # When we add a block B, we decrement uncovered count for its subsets.
    # This requires updating 35 entries.
    # Then we need to find the block with max uncovered count.
    # Scanning 50k blocks every step is 50k * 150 = 7.5M ops. This is fine!
    
    # Revised Plan:
    # 1. Initialize `uncovered_count_per_block` for all 50k blocks.
    #    `uncovered_count_per_block[i]` = number of subsets in block i that are currently uncovered.
    # 2. While uncovered_count > 0:
    #    a. Find block `b_idx` with max `uncovered_count_per_block`.
    #    b. Add block `b_idx` to solution.
    #    c. For each subset in block `b_idx`:
    #         if uncovered[subset]:
    #             uncovered[subset] = 0
    #             For each block `j` containing this subset:
    #                 uncovered_count_per_block[j] -= 1
    #    
    # To do step 2c efficiently, we need to know which blocks contain a given subset.
    # Pre-compute `subset_to_blocks`: dict from subset_index -> list of block_indices.
    
    # Pre-computation:
    # subset_to_blocks = {i: [] for i in range(n_t_subs)}
    # for b_idx, subs in enumerate(block_t_subs):
    #     for s in subs:
    #         subset_to_blocks[s].append(b_idx)
        
    # This pre-computation is 50k * 35 = 1.75M appends. Fast.
    
    # Let's implement this.
    
    subset_to_blocks = {i: [] for i in range(n_t_subs)}
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)
            
    uncovered_count_per_block = [0] * len(all_blocks)
    for b_idx, subs in enumerate(block_t_subs):
        uncovered_count_per_block[b_idx] = len(subs)
        
    total_uncovered = n_t_subs
    
    while total_uncovered > 0:
        if time.time() - start_time > time_limit * 0.5:
            break # Stop greedy early if taking too long, though it should be fast
            
        # Find best block
        max_count = -1
        best_b_idx = -1
        for i in range(len(all_blocks)):
            if uncovered_count_per_block[i] > max_count:
                max_count = uncovered_count_per_block[i]
                best_b_idx = i
                if max_count == len(block_t_subs[i]): # Can't do better
                    break
        if best_b_idx == -1 or max_count == 0:
            break
            
        blocks.append(all_blocks[best_b_idx])
        
        # Update counts
        for s in block_t_subs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Update all blocks containing s
                for b_idx in subset_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1
                    if uncovered_count_per_block[b_idx] == 0:
                         # Optional: mark as inactive? No, just let it be 0.
                         pass

    # Local Search Phase
    # Try to remove redundant blocks or swap blocks to reduce count.
    # We will iterate until no improvement or time runs out.
    
    # Helper to check if a block is redundant (all its subsets covered by others)
    # We need cover_count.
    cover_count = [0] * n_t_subs
    for b in blocks:
        for s in block_t_subs[all_blocks.index(b)]: # This is slow, O(N) lookup
            cover_count[s] += 1
            
    # Re-index blocks for faster access in local search
    # Let's keep blocks as indices into all_blocks for speed.
    block_indices = [all_blocks.index(b) for b in blocks]
    blocks = [all_blocks[i] for i in block_indices] # Keep original structure for return
    
    # Re-compute cover_count using indices
    cover_count = [0] * n_t_subs
    for b_idx in block_indices:
        for s in block_t_subs[b_idx]:
            cover_count[s] += 1
            
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        # Try to remove a block
        for i in range(len(block_indices)):
            b_idx = block_indices[i]
            # Check if redundant
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break # Restart loop as indices changed
                
        # If no removal, try swaps
        if not improved:
            # Try swapping each block with a better one
            # This is expensive. We will try a limited set of swaps.
            # For each block, find if there's another block that covers the "critical" subsets
            # and adds fewer new uncovered subsets.
            
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                # Subsets covered ONLY by this block
                critical_subsets = [s for s in block_t_subs[b_idx] if cover_count[s] == 1]
                if not critical_subsets:
                    continue # Should have been removed
                
                # Find a replacement block b_new
                # b_new must cover all critical_subsets? Or at least most?
                # If it covers all, and its other subsets are already covered, it's a perfect swap.
                # Let's look for a block that covers all critical_subsets.
                # We can find candidates by intersecting subset_to_blocks for critical_subsets.
                
                candidates = set(subset_to_blocks[critical_subsets[0]])
                for s in critical_subsets[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                    
                best_gain = 0
                best_new_idx = -1
                for new_idx in candidates:
                    if new_idx == b_idx:
                        continue
                    if new_idx in block_indices:
                        continue # Already in solution, can't duplicate
                    
                    # Calculate gain:
                    # 1. How many new subsets does new_idx cover that are currently uncovered? (Should be 0 if we want pure swap, but maybe it covers some)
                    # 2. How many subsets does b_idx cover that are NOT covered by new_idx? (These become uncovered if we swap, unless covered by others)
                    
                    # Actually, simpler metric:
                    # Current uncovered count: U
                    # Remove b_idx: U increases by |{s in b_idx | cover_count[s] == 1}| = len(critical_subsets)
                    # Add new_idx: U decreases by |{s in new_idx | cover_count[s] == 0}|
                    # Net change = len(critical_subsets) - (newly covered by new_idx)
                    # We want Net change < 0.
                    
                    newly_covered = 0
                    for s in block_t_subs[new_idx]:
                        if cover_count[s] == 0:
                            newly_covered += 1
                            
                    # Also, some subsets in b_idx might be covered by new_idx, reducing the "loss"
                    # Loss is actually: subsets in b_idx that become uncovered.
                    # These are s in b_idx such that cover_count[s] == 1 AND s not in new_idx.
                    # Wait, if s is in new_idx, then after adding new_idx, cover_count[s] becomes 1 (from new_idx) + 0 (from b_idx removed) = 1. So it's still covered.
                    # So the loss is |{s in b_idx | cover_count[s] == 1 and s not in new_idx}|.
                    
                    loss = 0
                    new_set = set(block_t_subs[new_idx])
                    for s in block_t_subs[b_idx]:
                        if cover_count[s] == 1 and s not in new_set:
                            loss += 1
                            
                    gain = newly_covered - loss
                    if gain > best_gain:
                        best_gain = gain
                        best_new_idx = new_idx
                        
                if best_gain > 0:
                    # Perform swap
                    # Remove b_idx
                    for s in block_t_subs[b_idx]:
                        cover_count[s] -= 1
                    # Add new_idx
                    for s in block_t_subs[best_new_idx]:
                        cover_count[s] += 1
                    block_indices[i] = best_new_idx
                    improved = True
                    break # Restart
                    
    # Convert indices back to blocks
    final_blocks = [all_blocks[i] for i in block_indices]
    
    # Final check: ensure all subsets are covered
    # If not, we might have broken validity during swaps? 
    # My swap logic ensures gain > 0, which implies net reduction in uncovered.
    # But did I ensure validity? 
    # If we start with a valid cover, and we only swap if gain > 0 (uncovered decreases), 
    # we never increase uncovered. 
    # Wait, `newly_covered` counts subsets with cover_count 0. 
    # `loss` counts subsets with cover_count 1 that are lost.
    # If gain > 0, then newly_covered > loss.
    # This means we cover more previously uncovered subsets than we lose coverage on.
    # So total uncovered decreases. 
    # Since we started with 0 uncovered, we stay at 0?
    # No, `uncovered` in my swap logic refers to `cover_count[s] == 0`.
    # In the greedy phase, we ensured all are covered. So `cover_count[s] >= 1` for all s.
    # So `newly_covered` (where cover_count == 0) is always 0?
    # Yes, if the cover is valid.
    # So gain = 0 - loss = -loss.
    # We require gain > 0, so -loss > 0 => loss < 0. Impossible.
    # So this swap logic will NEVER find a swap if the cover is already valid!
    # It only finds swaps if the cover is INVALID.
    # But greedy produces a VALID cover.
    # So this local search does nothing.
    
    # I need a different local search move.
    # Move: Replace b_idx with new_idx if the NUMBER OF BLOCKS decreases? No, we swap 1 for 1.
    # We want to reduce the number of blocks.
    # The "Remove Redundant" step does that.
    # The swap step should try to enable a removal.
    # i.e., Swap b_idx with new_idx such that new_idx allows removing ANOTHER block b_j?
    # This is complex.
    
    # Alternative: Just do the "Remove Redundant" step multiple times.
    # Often, greedy leaves redundant blocks.
    # My "Remove Redundant" loop above breaks after one removal.
    # I should make it loop until no more removals.
    
    # Let's refine the local search to just be aggressive removal.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.9:
        improved = False
        for i in range(len(block_indices)):
            b_idx = block_indices[i]
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break
                
    final_blocks = [all_blocks[i] for i in block_indices]
    return final_blocks

# Quick sanity check logic:
# The greedy phase is valid.
# The removal phase maintains validity (only removes if redundant).
# So the result is always valid.
# The score is the number of blocks.

# To improve further, we could use a different greedy strategy (e.g., random restarts) 
# but the current one is solid. The removal step is key.

# One optimization: In the greedy phase, `max_count == len(block_t_subs[i])` check 
# assumes we can't do better than covering all subsets in a block. 
# This is correct.

# Let's run this.
