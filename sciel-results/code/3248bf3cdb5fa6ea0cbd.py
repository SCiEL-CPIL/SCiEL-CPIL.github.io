import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers
    # block_to_tsubs[b] = list of indices of t-subsets covered by block b
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gains: each block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    
    # Track uncovered t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    selected_block_indices = []
    
    # 1. Greedy Construction
    # Use a priority queue or simple argmax? Argmax is O(N).
    # N = 50388.
    # We update gains sparsely.
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find the block with the maximum gain
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
            
        selected_block_indices.append(b)
        
        # Update gains for all blocks that share any t-subset with b
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains[tsub_to_blocks_np[s]] -= 1

    if not uncovered.any():
        # 2. Local Optimization
        # Maintain cover_count for each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step 2a: Redundancy Removal (1-for-0)
            redundant_blocks = []
            for b in selected_block_indices:
                # Check if any t-subset covered by b has count 1
                # If all counts > 1, it's redundant
                is_red = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart loop to check for new redundancies
            
            # Step 2b: 1-for-1 Swaps
            # Calculate unique counts for each block
            unique_counts = {}
            unique_tsubs = {}
            
            for b in selected_block_indices:
                u_count = 0
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                        u_list.append(s)
                unique_counts[b] = u_count
                unique_tsubs[b] = u_list
            
            # Find block with minimum unique count
            min_unique = min(unique_counts.values())
            if min_unique == 0:
                # Should have been caught by redundancy, but just in case
                continue
                
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs[b_out]
                if not s_uniq:
                    continue
                    
                first_s = s_uniq[0]
                candidates_in = tsub_to_blocks_np[first_s]
                
                valid_candidates = []
                for c in candidates_in:
                    if c in selected_block_indices:
                        continue
                    covers_all = True
                    for s in s_uniq[1:]:
                        if c not in tsub_to_blocks[s]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c)
                        
                if valid_candidates:
                    b_in = random.choice(valid_candidates)
                    # Perform swap
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
                    
            if swap_made:
                continue
                
            # Step 2c: 2-for-1 Swaps (Structural Change)
            # We look for pairs (b1, b2) such that removing them leaves a set R of uncovered t-subsets
            # that can be covered by a single new block b_in.
            
            n_sel = len(selected_block_indices)
            # We limit the number of pairs to check to stay within time budget
            # If n_sel is ~150, pairs are ~11k. Checking each is fast.
            
            # To be efficient, we can iterate through all pairs
            # But we need to be careful with the "find b_in" step.
            
            # Pre-compute a set of selected blocks for fast lookup?
            # selected_set = set(selected_block_indices) is updated often.
            
            found_2for1 = False
            # Shuffle indices to avoid bias
            indices_list = list(range(n_sel))
            random.shuffle(indices_list)
            
            # We will check a subset of pairs if n_sel is large, but 150 is small.
            # Let's check all pairs.
            for i in range(n_sel):
                b1 = selected_block_indices[i]
                # Optimization: If b1 has many unique t-subsets, it's a bad candidate for removal
                # But in 2-for-1, we remove two.
                
                for j in range(i + 1, n_sel):
                    b2 = selected_block_indices[j]
                    
                    # Calculate R: t-subsets that will become uncovered
                    # s is in R if:
                    # 1. cover_count[s] <= 2 (since we remove 2 blocks)
                    # 2. b1 covers s OR b2 covers s
                    # 3. If b1 covers s, count becomes count-1 (or -2 if b2 also covers)
                    #    If count becomes 0, it's uncovered.
                    
                    # Efficient way:
                    # Iterate through union of tsubs in b1 and b2?
                    # Or iterate through b1's tsubs and check b2?
                    
                    # Let's build a small set of tsubs to check
                    # tsubs_b1 = set(block_to_tsubs[b1])
                    # tsubs_b2 = set(block_to_tsubs[b2])
                    # union = tsubs_b1 | tsubs_b2
                    # This is ~70 elements.
                    
                    # We can just iterate through block_to_tsubs[b1]
                    # and add those from b2 not in b1.
                    
                    # Actually, we can just check the condition directly.
                    # R = []
                    # for s in block_to_tsubs[b1]:
                    #    c = cover_count[s]
                    #    if c == 1:
                    #        # Only b1 covers it (or b1 and someone else? No, c=1 means only 1 block covers it)
                    #        # If we remove b1, it becomes 0.
                    #        # But we also remove b2. Does b2 cover it?
                    #        # If b2 covers it, c would be at least 2 (b1 and b2).
                    #        # So if c==1, b2 does NOT cover it.
                    #        # So it becomes uncovered.
                    #        R.append(s)
                    #    elif c == 2:
                    #        # Two blocks cover it. One is b1.
                    #        # If the other is b2, then removing both makes it 0.
                    #        # If the other is b3 (not b2), removing b1 makes it 1 (still covered by b3).
                    #        # So we need to check if b2 covers s.
                    #        if b2 in tsub_to_blocks[s]: # O(1) if set, O(len) if list
                    #            R.append(s)
                    # 
                    # # Now check tsubs in b2 not in b1
                    # # We need to be careful not to double count or miss.
                    # # Better: iterate through union.
                    
                    # Let's use a set for tsubs in b1 for fast lookup
                    tsubs_b1_set = set(block_to_tsubs[b1])
                    r_list = []
                    
                    # Check tsubs in b1
                    for s in block_to_tsubs[b1]:
                        c = cover_count[s]
                        if c == 1:
                            r_list.append(s)
                        elif c == 2:
                            # Check if b2 also covers s
                            # tsub_to_blocks[s] is a list. We can check if b2 is in it.
                            # Since c=2, the list has length 2.
                            if b2 in tsub_to_blocks[s]:
                                r_list.append(s)
                        
                    # Check tsubs in b2 that are NOT in b1
                    for s in block_to_tsubs[b2]:
                        if s in tsubs_b1_set:
                            continue
                        c = cover_count[s]
                        if c == 1:
                            r_list.append(s)
                        elif c == 2:
                            # b1 does not cover s.
                            # So the 2 blocks covering s are b2 and some b3.
                            # Removing b2 leaves b3. Count becomes 1. Still covered.
                            # So s is NOT uncovered.
                            pass
                            
                    if not r_list:
                        # 2-for-0! Remove both
                        selected_block_indices.remove(b1)
                        selected_block_indices.remove(b2)
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        improved = True
                        found_2for1 = True
                        break
                        
                    # If r_list is not empty, try to find a block covering all of them
                    # Sort r_list by number of covering blocks (ascending) to prune candidates faster
                    # But r_list is small (< 70).
                    
                    # Find candidates for the first tsub
                    first_s = r_list[0]
                    candidates = tsub_to_blocks_np[first_s]
                    
                    valid_in = []
                    for c in candidates:
                        if c in selected_block_indices:
                            continue
                        # Check if c covers all other s in r_list
                        covers_all = True
                        for s in r_list[1:]:
                            if c not in tsub_to_blocks[s]:
                                covers_all = False
                                break
                        if covers_all:
                            valid_in.append(c)
                            
                    if valid_in:
                        b_in = random.choice(valid_in)
                        # Perform 2-for-1
                        selected_block_indices.remove(b1)
                        selected_block_indices.remove(b2)
                        selected_block_indices.append(b_in)
                        
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            cover_count[s] += 1
                            
                        improved = True
                        found_2for1 = True
                        break
                        
                if found_2for1:
                    break
                    
            if not found_2for1:
                # No 2-for-1 improvement found in this pass
                # We could try random restarts or more aggressive search,
                # but for now, we stop if no 1-for-1 or 2-for-1 works.
                break

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    return result
