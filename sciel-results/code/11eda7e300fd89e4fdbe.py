import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {block: i for i, block in enumerate(all_k_blocks)}
    num_k_blocks = len(all_k_blocks)
    
    # coverage[block_idx] -> list of t_subset_idx covered
    block_coverage = [None] * num_k_blocks
    for i, block in enumerate(all_k_blocks):
        covered = []
        for t_sub in itertools.combinations(block, t):
            covered.append(t_subset_to_idx[t_sub])
        block_coverage[i] = covered
        
    # Greedy Initialization
    coverage_count = [0] * num_t_subsets
    active_blocks = [] # List of block indices
    
    # Use a list of uncovered indices for faster checking? 
    # No, checking coverage_count[t] == 0 is O(1).
    # We need to track the number of uncovered subsets to know when to stop.
    num_uncovered = num_t_subsets
    
    while num_uncovered > 0:
        if time.time() > deadline - 20:
            break # Leave time for local search
            
        best_block = -1
        best_gain = -1
        
        # Optimization: Randomly sample a subset of blocks if we are running low on time?
        # For now, full scan.
        
        for b_idx in range(num_k_blocks):
            gain = 0
            for t_idx in block_coverage[b_idx]:
                if coverage_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1 or best_gain == 0:
            break # Should not happen if num_uncovered > 0
            
        active_blocks.append(best_block)
        for t_idx in block_coverage[best_block]:
            if coverage_count[t_idx] == 0:
                num_uncovered -= 1
            coverage_count[t_idx] += 1
                
    # Convert to list of blocks (tuples)
    current_blocks = [all_k_blocks[i] for i in active_blocks]
    
    # Local Search: Redundancy Removal
    # We work with a set of active block indices for O(1) removal
    active_block_set = set(active_blocks)
    
    # Re-initialize coverage counts from active_block_set to be sure
    coverage_count = [0] * num_t_subsets
    for b_idx in active_block_set:
        for t_idx in block_coverage[b_idx]:
            coverage_count[t_idx] += 1
            
    improved = True
    while time.time() < deadline - 5 and improved:
        improved = False
        
        # 1. Try to remove redundant blocks
        # A block is redundant if for all t in block_coverage[b], coverage_count[t] > 1
        blocks_to_check = list(active_block_set)
        random.shuffle(blocks_to_check) # Random order to escape local minima
        
        for b_idx in blocks_to_check:
            if time.time() > deadline - 5:
                break
            
            is_redundant = True
            for t_idx in block_coverage[b_idx]:
                if coverage_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove b_idx
                for t_idx in block_coverage[b_idx]:
                    coverage_count[t_idx] -= 1
                active_block_set.remove(b_idx)
                improved = True
        
        # 2. Try to remove pairs of blocks that are jointly redundant
        # If removing b1 and b2 results in no uncovered subsets, remove them.
        # Condition: For all t in cov(b1) U cov(b2), 
        #   remaining_count = count(t) - (1 if b1 covers t else 0) - (1 if b2 covers t else 0)
        #   must be > 0.
        
        if len(active_block_set) > 1:
            blocks_list = list(active_block_set)
            random.shuffle(blocks_list)
            
            # Limit number of pairs checked per iteration
            max_pairs = min(len(blocks_list) * (len(blocks_list) - 1) // 2, 5000)
            pairs_checked = 0
            
            for i in range(len(blocks_list)):
                if pairs_checked >= max_pairs:
                    break
                b1 = blocks_list[i]
                for j in range(i + 1, len(blocks_list)):
                    if pairs_checked >= max_pairs:
                        break
                    b2 = blocks_list[j]
                    pairs_checked += 1
                    
                    # Check if joint removal is safe
                    safe = True
                    
                    # Helper to check if b2 covers t_idx
                    # Since lists are small, linear search is ok, but set is better?
                    # We don't have sets for block_coverage.
                    # We can just check membership.
                    
                    # Check subsets covered by b1
                    for t_idx in block_coverage[b1]:
                        c = coverage_count[t_idx]
                        # b1 covers t_idx, so subtract 1
                        # Does b2 cover t_idx?
                        if t_idx in block_coverage[b2]:
                            # Subtract another 1
                            if c == 2:
                                safe = False
                                break
                        else:
                            if c == 1:
                                safe = False
                                break
                        # If c > 2, safe.
                        # If c == 2 and b2 covers, unsafe.
                        # If c == 1 and b2 doesn't cover, unsafe.
                        # If c == 2 and b2 doesn't cover, safe (remaining 1).
                        # If c == 1 and b2 covers, impossible (c would be >= 2).
                    
                    if not safe:
                        continue
                        
                    # Check subsets covered by b2 (that are NOT covered by b1)
                    # We can skip if we already checked t_idx in b1 loop.
                    # To be safe and simple, we check all in b2, but skip if already processed?
                    # Actually, if t is in both, we handled it in b1 loop.
                    # If t is only in b2, we need to check.
                    # We can just iterate b2 and check if t is in b1? 
                    # Or just check the condition again.
                    for t_idx in block_coverage[b2]:
                        # If t_idx was in b1, we already determined it's safe (or broke).
                        # If we broke, safe is False.
                        # If we didn't break, it was safe.
                        # So we only need to check t_idx NOT in b1.
                        # But checking "in b1" is O(12).
                        # Let's just re-evaluate for simplicity, it's fast enough.
                        c = coverage_count[t_idx]
                        if t_idx in block_coverage[b1]:
                            # Already checked in b1 loop. 
                            # But wait, if we broke out of b1 loop, safe is False.
                            # If we didn't break, it was safe.
                            # So we can skip.
                            pass
                        else:
                            # b2 covers t_idx, b1 does not.
                            if c == 1:
                                safe = False
                                break
                            # c >= 2 is safe (remaining >= 1).
                    
                    if safe:
                        # Remove b1 and b2
                        for t_idx in block_coverage[b1]:
                            coverage_count[t_idx] -= 1
                        for t_idx in block_coverage[b2]:
                            coverage_count[t_idx] -= 1
                        active_block_set.remove(b1)
                        active_block_set.remove(b2)
                        improved = True
                        break # Restart outer loop to update blocks_list
                    
                if improved:
                    break # Break inner loop to restart outer

    # Construct final output
    final_blocks = [list(all_k_blocks[i]) for i in active_block_set]
    
    # Fallback: If empty (should not happen), return greedy result?
    # We don't have it. But greedy should work.
    # If active_block_set is empty, return [] (which is invalid).
    # But for v=19, k=8, t=5, greedy will find blocks.
    
    return final_blocks
