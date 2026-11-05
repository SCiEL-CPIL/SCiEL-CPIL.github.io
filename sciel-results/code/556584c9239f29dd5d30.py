import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute for each point, the list of subset IDs it belongs to
    subsets_by_point = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subsets_by_point[x].append(i)
            
    # Precompute for each block (k-subset), the list of subset IDs it covers
    # This is expensive if we do it for all k-subsets (50388 * 7 choose 4... wait, 7 choose 4 is 35).
    # Total operations ~ 50388 * 35 ~ 1.7M. This is feasible.
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = []
    for b in all_blocks:
        b_set = set(b)
        subs = []
        for s in all_subsets:
            if all(x in b_set for x in s):
                subs.append(subset_to_id[s])
        block_to_subsets.append(subs)
        
    # Map block tuple to index in all_blocks for quick lookup?
    # Not strictly needed if we just use indices.
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = [1] * num_subsets
        blocks = []
        # We will store block indices
        block_indices = []
        
        while any(uncovered):
            # Find the block that covers the most uncovered subsets
            best_block_idx = -1
            best_count = 0
            # Optimization: We can't iterate all 50k blocks every step if we have many steps.
            # But max blocks is ~150. 150 * 50k = 7.5M operations. Feasible.
            # We can speed this up by only considering blocks that cover at least one uncovered subset.
            
            # Better: Iterate through uncovered subsets, pick one, look at blocks covering it?
            # Let's stick to iterating all blocks but break early if we find a perfect one? No.
            
            # Efficient Greedy:
            # 1. Pick a random uncovered subset S.
            # 2. Find all blocks that cover S.
            # 3. Among those, pick the one that covers the most OTHER uncovered subsets.
            
            # We need a mapping: subset_id -> list of block_indices that cover it.
            # Let's build this on the fly or precompute?
            # Precompute: subset_to_blocks
            # This is the transpose of block_to_subsets.
            
            # Since we are in a function, let's build it once outside if possible, 
            # but we need to pass it or rebuild. 
            # Rebuilding 50k * 35 is fast.
            
            # Let's just do the simple O(N_blocks * N_subsets_in_block) check for now, 
            # but restricted to blocks that intersect uncovered.
            
            # Actually, the "Pick random uncovered subset" strategy is faster.
            uncov_indices = [i for i, c in enumerate(uncovered) if c > 0]
            if not uncov_indices:
                break
            target_subset_idx = random.choice(uncov_indices)
            target_subset = all_subsets[target_subset_idx]
            
            # Find blocks containing target_subset
            candidates = []
            for b_idx, b in enumerate(all_blocks):
                if all(x in b for x in target_subset):
                    candidates.append(b_idx)
            
            if not candidates:
                # Should not happen
                break
            
            # Evaluate candidates
            best_cand = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for s_idx in block_to_subsets[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = b_idx
            
            if best_cand == -1:
                break
                
            # Add block
            block_indices.append(best_cand)
            for s_idx in block_to_subsets[best_cand]:
                uncovered[s_idx] -= 1 # Actually we just need to know if it's 0 or >0. 
                # Let's use a set of uncovered indices for faster checking?
                # No, array is fine.
            
            # Optimization: If uncovered count becomes 0, break.
            # We can maintain a count of uncovered.
            
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in block_indices]

    # This greedy is O(Steps * Candidates * SubsetsPerBlock).
    # Steps ~ 150. Candidates ~ ? 
    # For a specific 4-subset in 19 points, how many 7-blocks contain it?
    # We need to choose 3 more points from the remaining 15. C(15,3) = 455.
    # So candidates ~ 455.
    # SubsetsPerBlock = C(7,4) = 35.
    # Cost per step ~ 455 * 35 = 15k ops.
    # 150 steps * 15k = 2.2M ops. Very fast.
    
    # Now, Local Search
    def local_search(blocks, time_limit=10):
        # blocks: list of lists
        # Convert to indices for faster manipulation?
        # Let's work with indices into all_blocks.
        
        # Map block tuple to index
        block_map = {tuple(b): i for i, b in enumerate(all_blocks)}
        
        current_blocks_idx = [block_map[tuple(b)] for b in blocks]
        
        # Precompute cover counts
        cover_count = [0] * num_subsets
        for b_idx in current_blocks_idx:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
        
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            # A block is redundant if for all s in block, cover_count[s] > 1
            for i in range(len(current_blocks_idx)):
                b_idx = current_blocks_idx[i]
                redundant = True
                for s_idx in block_to_subsets[b_idx]:
                    if cover_count[s_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count[s_idx] -= 1
                    current_blocks_idx.pop(i)
                    improved = True
                    break # Restart loop to re-evaluate
            
            # 2. Try Swaps to enable removal
            # Pick a random block to swap
            if len(current_blocks_idx) > 0:
                swap_idx = random.randint(0, len(current_blocks_idx) - 1)
                old_b_idx = current_blocks_idx[swap_idx]
                
                # Generate a new block
                # To make it effective, pick a block that covers many "critical" subsets?
                # Or just random?
                # Let's try a few random swaps.
                
                # Simulate removal of old_b_idx
                # We need to check if any other block becomes redundant after swapping.
                
                # Tentative state:
                # Remove old_b_idx
                temp_cover = cover_count.copy()
                for s_idx in block_to_subsets[old_b_idx]:
                    temp_cover[s_idx] -= 1
                
                # Try a few random new blocks
                # We want a new block that covers subsets that are now at risk (count 1) 
                # or just generally useful.
                
                # Strategy: Pick a random uncovered or low-count subset, find blocks covering it.
                # Actually, just pick a random block index.
                
                found_swap = False
                for _ in range(20): # Try 20 random swaps
                    new_b_idx = random.randint(0, len(all_blocks) - 1)
                    if new_b_idx == old_b_idx:
                        continue
                    
                    # Check if adding new_b_idx to (current - old) creates a redundancy
                    # We already have temp_cover (which is current - old)
                    # We add new_b_idx to temp_cover
                    # Then check if any block in (current - old - new?) is redundant.
                    
                    # Wait, the new set is (current_blocks - {old}) + {new}
                    # Let's call this new_set.
                    # We want to know if any block in new_set is redundant.
                    # Specifically, if we can remove ANOTHER block from new_set, we gain 1.
                    
                    # Update temp_cover with new_b_idx
                    for s_idx in block_to_subsets[new_b_idx]:
                        temp_cover[s_idx] += 1
                        
                    # Check for redundancy in the new set
                    # The new set consists of all blocks in current_blocks_idx except old_b_idx, plus new_b_idx.
                    # We iterate through indices in current_blocks_idx (excluding swap_idx) and check redundancy.
                    # Also check new_b_idx itself.
                    
                    # Optimization: Only check blocks that were "affected" or likely to be redundant.
                    # Actually, a block is redundant if all its subsets have count > 1 in the NEW cover.
                    
                    # Let's check if new_b_idx is redundant (unlikely if we just added it, unless it covers nothing new? No, it covers something).
                    # Check if any existing block is redundant.
                    
                    # To speed up, we can just check a random sample of blocks?
                    # No, we need to be sure.
                    # But checking all blocks is O(N_blocks * 35). 150 * 35 = 5k ops.
                    # 20 swaps * 5k = 100k ops. Fast.
                    
                    redundant_found = -1
                    for j, b_idx in enumerate(current_blocks_idx):
                        if j == swap_idx:
                            continue
                        # Check if b_idx is redundant in the new cover
                        is_red = True
                        for s_idx in block_to_subsets[b_idx]:
                            if temp_cover[s_idx] <= 1:
                                is_red = False
                                break
                        if is_red:
                            redundant_found = j
                            break
                    
                    # Also check if new_b_idx is redundant (shouldn't be, but just in case)
                    if redundant_found == -1:
                        is_red_new = True
                        for s_idx in block_to_subsets[new_b_idx]:
                            if temp_cover[s_idx] <= 1:
                                is_red_new = False
                                break
                        if is_red_new:
                            redundant_found = -2 # Special marker for new block? No, we don't want to remove the block we just added.
                            # Actually, if new block is redundant, the swap is pointless (size same, no removal).
                            # So we only care if an EXISTING block becomes redundant.
                    
                    if redundant_found != -1 and redundant_found != -2:
                        # Accept swap and remove redundant block
                        # 1. Remove old_b_idx from current_blocks_idx
                        # 2. Add new_b_idx
                        # 3. Remove redundant block at redundant_found
                        
                        # Update cover_count
                        # We already have temp_cover which is (current - old) + new
                        # We need to remove the redundant block from temp_cover to get final cover_count
                        red_b_idx = current_blocks_idx[redundant_found]
                        for s_idx in block_to_subsets[red_b_idx]:
                            temp_cover[s_idx] -= 1
                        
                        cover_count = temp_cover
                        
                        # Update list
                        # Remove old
                        current_blocks_idx[swap_idx] = new_b_idx
                        # Remove redundant
                        # Note: indices might have shifted if we pop.
                        # It's safer to rebuild the list.
                        new_list = []
                        for j, b_idx in enumerate(current_blocks_idx):
                            if j == redundant_found:
                                continue
                            new_list.append(b_idx)
                        current_blocks_idx = new_list
                        
                        improved = True
                        found_swap = True
                        break
                    
                    # Rollback temp_cover for next trial
                    for s_idx in block_to_subsets[new_b_idx]:
                        temp_cover[s_idx] -= 1
                    # temp_cover is now back to (current - old)
                    
                if found_swap:
                    break # Restart outer loop
            
        return [list(all_blocks[i]) for i in current_blocks_idx]

    # Main Loop
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 110
    
    # Run multiple restarts
    run_id = 0
    while time.time() - start_time < time_limit:
        # Greedy phase
        g_start = time.time()
        blocks = greedy_once(run_id)
        g_time = time.time() - g_start
        
        # Local Search phase
        # Allocate remaining time to local search
        # If greedy took 0.1s, we have ~110s.
        # Let's give local search up to 5 seconds per run?
        # Or just run until time limit?
        # If we do many restarts, we should limit local search time per restart.
        
        ls_time_limit = min(5.0, time_limit - (time.time() - start_time))
        if ls_time_limit > 0.5:
            blocks = local_search(blocks, ls_time_limit)
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        run_id += 1
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break

    return best_blocks
