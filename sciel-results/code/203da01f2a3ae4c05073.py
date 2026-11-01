import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(points, k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: block_idx -> list of t_subset_idx
    block_to_tsubs = []
    # Also store as a set for fast lookup in local search
    block_to_tsubs_set = []
    
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse: t_subset_idx -> list of block_idx
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for potential vectorized ops, but keep lists for iteration speed
    # tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_construction(randomize=False):
        uncovered = np.ones(num_tsubs, dtype=bool)
        # Initial gain for each block is number of t-subsets it covers
        gains = np.array([len(b) for b in block_to_tsubs], dtype=np.int32)
        
        selected_block_indices = []
        selected_set = set()
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if randomize and random.random() < 0.2:
                # Randomly pick from top 5% of gains
                threshold = np.percentile(gains[gains > 0], 95) if np.any(gains > 0) else 0
                candidates = np.where((gains >= threshold) & (gains > 0))[0]
                if len(candidates) == 0:
                    b = int(np.argmax(gains))
                else:
                    b = int(random.choice(candidates))
            else:
                b = int(np.argmax(gains))
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            selected_set.add(b)
            
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Update gains for all blocks covering this t-subset
                    for other_b in tsub_to_blocks[s_idx]:
                        gains[other_b] -= 1
                    
        if uncovered.any():
            return None
            
        return selected_block_indices

    def local_optimize(blocks, max_time=10.0):
        if not blocks:
            return []
            
        selected_list = list(blocks)
        selected_set = set(selected_list)
        
        # cover_count[s] = number of selected blocks covering t-subset s
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            # A block is redundant if all t-subsets it covers are covered by at least one other block
            # i.e., for all s in block, cover_count[s] > 1
            # We can check this quickly
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True

        def try_remove_redundant():
            removed = False
            # Iterate until no more removals
            while True:
                if time.time() > start_time + max_time:
                    break
                to_remove = []
                for b in selected_list:
                    if is_redundant(b):
                        to_remove.append(b)
                if not to_remove:
                    break
                for b in to_remove:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                removed = True
            return removed

        def try_1_for_1_swap():
            if time.time() > start_time + max_time:
                return False
            
            # Identify blocks with minimal unique coverage
            # unique_tsubs[b] = list of s where cover_count[s] == 1 and b covers s
            # We want to swap out a block with few unique t-subsets
            # and find a block that covers those unique t-subsets and is not in the set.
            
            # To speed up, we only consider swapping out blocks that have unique coverage > 0?
            # Actually, if unique coverage is 0, it's redundant (handled above).
            # So all blocks have unique coverage >= 1.
            
            # Sort by number of unique t-subsets
            unique_counts = {}
            unique_lists = {}
            for b in selected_list:
                ulist = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        ulist.append(s)
                unique_counts[b] = len(ulist)
                unique_lists[b] = ulist
            
            # Try swapping out the block with the smallest unique count first (easiest to replace?)
            # Or random order? Let's try smallest first as it's "cheapest" to move.
            sorted_blocks = sorted(selected_list, key=lambda x: unique_counts[x])
            
            for b_out in sorted_blocks:
                if time.time() > start_time + max_time:
                    return False
                    
                u_list = unique_lists[b_out]
                if not u_list:
                    continue
                    
                # Find a candidate block that covers all s in u_list
                # The candidate must be in the intersection of tsub_to_blocks[s] for all s in u_list
                # Since u_list is small (usually), we can take the intersection of the lists.
                
                # Optimize: pick the s with the smallest tsub_to_blocks list
                s_min = min(u_list, key=lambda s: len(tsub_to_blocks[s]))
                candidates = tsub_to_blocks[s_min]
                
                found = False
                for c in candidates:
                    if c in selected_set:
                        continue
                    
                    # Check if c covers all other s in u_list
                    c_set = block_to_tsubs_set[c]
                    valid = True
                    for s in u_list:
                        if s not in c_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        for s in block_to_tsubs[b_out]:
                            cover_count[s] -= 1
                        
                        selected_list.append(c)
                        selected_set.add(c)
                        for s in block_to_tsubs[c]:
                            cover_count[s] += 1
                        
                        return True
            return False

        def try_2_for_2_swap():
            if time.time() > start_time + max_time:
                return False
                
            # This is more complex.
            # We look for a pair of blocks (b1, b2) that can be replaced by (c1, c2).
            # The union of unique t-subsets covered by b1 and b2 must be covered by c1 U c2.
            
            # Heuristic: 
            # 1. Identify "weak" blocks (low unique count).
            # 2. Pair them up.
            # 3. Try to find replacements.
            
            unique_counts = {}
            unique_lists = {}
            for b in selected_list:
                ulist = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        ulist.append(s)
                unique_counts[b] = len(ulist)
                unique_lists[b] = ulist
            
            # Only consider blocks with small unique counts for swapping out
            # Threshold: maybe <= 10 unique t-subsets
            candidates_out = [b for b in selected_list if unique_counts[b] <= 10]
            
            if len(candidates_out) < 2:
                return False
                
            # Shuffle to avoid bias
            random.shuffle(candidates_out)
            
            # Try a few pairs
            max_pairs_to_try = 20
            pairs_tried = 0
            
            for i in range(len(candidates_out)):
                for j in range(i+1, len(candidates_out)):
                    if pairs_tried >= max_pairs_to_try:
                        break
                    if time.time() > start_time + max_time:
                        return False
                    pairs_tried += 1
                    
                    b1 = candidates_out[i]
                    b2 = candidates_out[j]
                    
                    # The set of t-subsets that MUST be covered by the new pair
                    # is the union of unique subsets of b1 and b2.
                    # Note: If a t-subset is covered by both b1 and b2, and cover_count is 2,
                    # it becomes 1 after removal. So it must be covered by c1 or c2.
                    # Actually, simpler logic:
                    # After removing b1 and b2, any t-subset s such that cover_count[s] <= 2 
                    # and covered by b1 or b2 must be covered by c1 or c2.
                    
                    # Let's construct the "required" set.
                    required = set()
                    for s in block_to_tsubs[b1]:
                        if cover_count[s] <= 2: # Will become 1 or 0
                            required.add(s)
                    for s in block_to_tsubs[b2]:
                        if cover_count[s] <= 2:
                            required.add(s)
                    
                    if not required:
                        continue
                        
                    # We need c1, c2 such that required subset of (c1 U c2).
                    # And c1, c2 not in selected_set (or one is, but we are swapping 2 for 2, so both new ones should ideally be new, or at least distinct from b1,b2).
                    # Actually, c1 and c2 must be distinct from b1 and b2.
                    
                    # Strategy:
                    # Pick a t-subset s in required.
                    # It must be covered by c1 or c2.
                    # Suppose c1 covers s.
                    # Then c2 must cover (required - c1).
                    
                    # To make this tractable, we can iterate over candidates for c1.
                    # Let's pick the s in required with the fewest covering blocks.
                    s_min = min(required, key=lambda s: len(tsub_to_blocks[s]))
                    c1_candidates = tsub_to_blocks[s_min]
                    
                    found = False
                    for c1 in c1_candidates:
                        if c1 in selected_set and c1 not in [b1, b2]:
                            # If c1 is already in the set, we are effectively doing a 1-for-1 swap on b1 or b2, 
                            # but we are treating it as 2-for-2. 
                            # If c1 is in the set, we are removing b1,b2 and adding c1 (which is already there, so no net change for c1) and c2.
                            # This reduces to finding c2 that covers required - c1.
                            # This is a valid move.
                            pass
                        if c1 in [b1, b2]:
                            continue
                            
                        c1_set = block_to_tsubs_set[c1]
                        remaining_required = required - c1_set
                        
                        if not remaining_required:
                            # c1 alone covers all required. 
                            # This means we can remove b1 and b2 and add c1. 
                            # This is a 2-for-1 reduction! 
                            # Wait, if we remove 2 and add 1, we reduce size by 1. 
                            # But we must ensure c1 doesn't cause issues. 
                            # Actually, if c1 covers all required, we can just add c1 and remove b1, b2.
                            # This is better than 2-for-2.
                            # Let's handle this case: 
                            # Remove b1, b2. Add c1.
                            # Check if valid.
                            # We need to ensure that no t-subset becomes uncovered.
                            # The 'required' set contained all s with count <= 2.
                            # Any s with count > 2 is safe.
                            # So if c1 covers all required, it's a valid 2-for-1.
                            
                            # Perform 2-for-1
                            selected_list.remove(b1)
                            selected_set.remove(b1)
                            for s in block_to_tsubs[b1]:
                                cover_count[s] -= 1
                                
                            selected_list.remove(b2)
                            selected_set.remove(b2)
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                                
                            if c1 not in selected_set:
                                selected_list.append(c1)
                                selected_set.add(c1)
                                for s in block_to_tsubs[c1]:
                                    cover_count[s] += 1
                            # If c1 was already in set, we just removed b1, b2.
                            
                            return True

                        # If remaining_required is not empty, we need c2 to cover it.
                        # c2 must cover all s in remaining_required.
                        # Pick s_min2 from remaining_required.
                        s_min2 = min(remaining_required, key=lambda s: len(tsub_to_blocks[s]))
                        c2_candidates = tsub_to_blocks[s_min2]
                        
                        for c2 in c2_candidates:
                            if c2 in [b1, b2, c1]:
                                continue
                            
                            c2_set = block_to_tsubs_set[c2]
                            # Check if c2 covers all remaining_required
                            valid = True
                            for s in remaining_required:
                                if s not in c2_set:
                                    valid = False
                                    break
                            
                            if valid:
                                # Perform 2-for-2 swap
                                selected_list.remove(b1)
                                selected_set.remove(b1)
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                    
                                selected_list.remove(b2)
                                selected_set.remove(b2)
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                    
                                if c1 not in selected_set:
                                    selected_list.append(c1)
                                    selected_set.add(c1)
                                    for s in block_to_tsubs[c1]:
                                        cover_count[s] += 1
                                else:
                                    # c1 was already in the set, so we didn't add it, but we didn't remove it either.
                                    # Wait, if c1 was in the set, and we are "adding" it, we don't change its count.
                                    # But we removed b1, b2.
                                    # This logic holds.
                                    pass
                                    
                                if c2 not in selected_set:
                                    selected_list.append(c2)
                                    selected_set.add(c2)
                                    for s in block_to_tsubs[c2]:
                                        cover_count[s] += 1
                                else:
                                    pass
                                    
                                return True
                                
                    if found:
                        break
            return False

        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            # Cycle through optimization steps
            if try_remove_redundant():
                improved = True
            if try_1_for_1_swap():
                improved = True
            if try_2_for_2_swap():
                improved = True
                
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        random.seed(i)
        blocks = greedy_construction(randomize=(i > 0))
        
        if blocks is None:
            continue
        
        # Allocate remaining time to local optimization
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        if time_remaining < 2.0:
            # Not enough time for meaningful optimization
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_solution = blocks
            continue
            
        # Scale max_time for local_optimize based on remaining time
        # Leave 5s for final conversion
        opt_time = max(1.0, time_remaining - 5.0)
        
        optimized_blocks = local_optimize(blocks, max_time=opt_time)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
        if best_len <= 151:
            break

    if best_solution is None:
        # Fallback
        blocks = greedy_construction(randomize=False)
        if blocks:
            best_solution = local_optimize(blocks, max_time=5.0)
        else:
            return []
            
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result

