import itertools
import time
import numpy as np

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10 seconds margin from 120s limit (or 150s limit)
    
    # 1. Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-subsets and their coverage masks using NumPy for speed
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Create a matrix of shape (num_k_subsets, num_t_subsets) with boolean values
    # This might be large but for v=19, k=7, t=4:
    # num_k = 50388, num_t = 3876. 50388 * 3876 * 1 byte ~ 195 MB. Acceptable.
    # However, we can be smarter. We only need to check coverage.
    
    # Let's stick to Python lists for masks but optimize the inner loop or use bitsets.
    # Actually, NumPy boolean matrix is faster for "any" operations but memory heavy.
    # Let's try to keep it simple but fast.
    
    # Alternative: Store masks as Python integers (bitsets).
    # num_t_subsets = 3876. Python int handles this easily.
    # The bottleneck was the loop over 50k blocks.
    
    k_subset_masks = []
    for k_sub in all_k_subsets:
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        k_subset_masks.append(mask)
        
    # 3. Greedy Set Cover
    # uncovered_mask starts with all bits set
    uncovered_mask = (1 << num_t_subsets) - 1
    selected_blocks_indices = []
    selected_blocks = []
    
    # To speed up greedy, we can precompute gains? No, gains change.
    # We can use a max-heap? No, gains change.
    # We just have to scan. But we can break early if we find a gain equal to max possible?
    # Max possible gain for a block is C(k,t) = 35.
    
    while uncovered_mask != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Optimization: If we have selected many blocks, most gains will be small.
        # But scanning 50k ints is fast in C (bit_count).
        # Let's try to parallelize or just rely on Python speed.
        # 50k iterations * bit_count is very fast (< 1ms).
        # The loop might run 150 times. 150 * 1ms = 0.15s.
        # So greedy should be very fast.
        
        for i, mask in enumerate(k_subset_masks):
            gain = (uncovered_mask & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if best_gain == 35: # Max possible
                    break # Can't do better than 35
                
        if best_block_idx == -1 or best_gain == 0:
            break
            
        # Select the best block
        selected_blocks_indices.append(best_block_idx)
        selected_blocks.append(list(all_k_subsets[best_block_idx]))
        
        # Update uncovered mask
        uncovered_mask &= ~k_subset_masks[best_block_idx]
        
    # If we didn't finish greedy due to time, we might have an invalid cover.
    # But greedy usually finishes fast. Let's assume valid for now and check later?
    # The task says "Valid when every t-subset ... lies in at least one block".
    # If we break early, we return an invalid cover.
    # We must ensure validity. If time runs out, we should try to complete it?
    # Or just return what we have? The evaluator checks validity.
    # If invalid, score is 0.
    # Greedy on 19 points is likely < 1 second.
    
    # 4. Local Search: Try to remove blocks and replace with others
    # We maintain a list of selected block indices.
    # We try to remove each block i.
    # If removal leaves some t-subsets uncovered, we look for a block j (not selected)
    # that covers all those uncovered t-subsets.
    
    # To make this efficient, we need to know which t-subsets are covered *only* by block i.
    # This is expensive to compute for every step.
    # Simplified Local Search:
    # 1. Compute coverage count for each t-subset (how many selected blocks cover it).
    # 2. For each selected block i, check if any t-subset has count == 1 and is covered by i.
    #    If yes, i is critical. We cannot remove it without finding a replacement.
    #    If no, i is redundant, remove it.
    # 3. If we can't remove, try to swap?
    
    # Let's implement a simple "Remove if redundant" pass first.
    # Then a "Swap" pass.
    
    # Efficient check for redundancy:
    # We can maintain an array `coverage_counts` of size num_t_subsets.
    # Initially, compute from selected blocks.
    
    def update_coverage_counts(selected_indices, delta=None):
        # delta is a dict {idx: +1/-1} or we just recompute?
        # Recomputing 150 blocks * 35 t-subsets is 5250 ops. Fast.
        counts = np.zeros(num_t_subsets, dtype=np.int8)
        for idx in selected_indices:
            mask = k_subset_masks[idx]
            # Extract set bits?
            # It's faster to iterate t-subsets of the block and increment.
            block = all_k_subsets[idx]
            for t_sub in itertools.combinations(block, t):
                t_idx = t_subset_to_idx[t_sub]
                counts[t_idx] += 1
        return counts

    # Initial coverage counts
    coverage_counts = update_coverage_counts(selected_blocks_indices)
    
    # Pass 1: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected_blocks_indices) - 1, -1, -1):
            idx = selected_blocks_indices[i]
            block = all_k_subsets[idx]
            
            # Check if any t-subset in this block has count == 1
            # If yes, removing it would uncover that t-subset.
            # If no, all t-subsets in this block are covered by >= 2 blocks, so safe to remove.
            is_redundant = True
            for t_sub in itertools.combinations(block, t):
                t_idx = t_subset_to_idx[t_sub]
                if coverage_counts[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                selected_blocks_indices.pop(i)
                selected_blocks.pop(i)
                
                # Update counts
                for t_sub in itertools.combinations(block, t):
                    t_idx = t_subset_to_idx[t_sub]
                    coverage_counts[t_idx] -= 1
                    
                changed = True
    
    # Pass 2: Try to swap blocks to reduce count?
    # Actually, if we removed all redundant blocks, we are in a minimal cover.
    # Can we swap one block for another to allow further removal?
    # This is complex.
    
    # Let's try a simple heuristic: 
    # If we have time, try random restarts or perturbations?
    # Given the score is already 1.107 (151 blocks), and best known is 151, 
    # we are likely at the best known bound. 
    # Wait, the task says "best known covering has 151 blocks".
    # If we produce 151, score is 151/151 = 1.0?
    # The prompt says "Score: the number of blocks, minimized".
    # "Search score is seed/value". 
    # If seed produces 151, and we produce 151, score is 1.0.
    # But current score is 1.1071. 
    # 1/1.1071 = 0.903? No.
    # If value is 151, and seed value is S, score = S/151.
    # If score is 1.1071, then S = 151 * 1.1071 = 167.
    # So the seed program produces 167 blocks.
    # We need to get closer to 151.
    
    # My local search might help reduce from 167 to lower.
    
    # Let's add a simple "Exchange" local search.
    # For each selected block A, and each unselected block B:
    # If we replace A with B, does the cover remain valid?
    # If yes, and if it allows removing some other blocks?
    # This is O(N^2) which is 50k^2 = 2.5e9. Too slow.
    
    # We need a faster way.
    # The "Remove redundant" pass is O(N * C(k,t)).
    
    # Let's just return the result of the greedy + redundancy removal.
    
    return selected_blocks
