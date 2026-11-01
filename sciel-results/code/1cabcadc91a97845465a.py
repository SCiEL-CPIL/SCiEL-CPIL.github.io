import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # 1. Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)
    
    # 2. Pre-compute block -> t-subset IDs
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)
    
    # Store indices of t-subsets for each k-block
    # Shape: (num_k, C(k, t))
    # C(k, t) is fixed
    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)
    
    block_t_ids = np.zeros((num_k, num_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subs):
        block_set = set(block)
        for j, t_indices in enumerate(t_in_k):
            # Map local indices to global points
            global_t_sub = tuple(sorted(block[x] for x in t_indices))
            block_t_ids[i, j] = t_sub_map[global_t_sub]
            
    # 3. Greedy Cover
    covered = np.zeros(num_t, dtype=bool)
    used_blocks = set()
    solution = []
    remaining_indices = list(range(num_k))
    
    while not covered.all():
        if time.time() > deadline:
            break
            
        if not remaining_indices:
            break
            
        # Get indices of t-subsets for all remaining blocks
        # This creates a view or copy of shape (len(remaining), num_t_in_k)
        # We need to check which are uncovered
        # gains = sum of (not covered) for each block
        
        # Optimization: Instead of filtering remaining_indices list every time (O(N)),
        # we can just iterate all and mask out used ones?
        # But `remaining_indices` shrinks.
        
        # Let's get the sub-array of block_t_ids for remaining indices
        # This is a fancy index, creates a copy.
        # For 75k blocks, this is ~75k * 56 * 4 bytes = 16MB copy per step.
        # 300 steps -> 5GB memory traffic. Might be slow.
        
        # Alternative: Keep all blocks, mark used.
        # gains = np.sum(~covered[all_block_t_ids], axis=1)
        # Then set gains[used_blocks] = -1
        # This avoids the copy of the sub-array, just uses the full array.
        
        # Let's use the full array approach.
        # all_block_t_ids is (num_k, num_t_in_k)
        # ~covered[all_block_t_ids] is (num_k, num_t_in_k) boolean
        # sum(axis=1) is (num_k,)
        
        # To avoid recalculating for used blocks, we can just set their gain to -1 after calculation?
        # Or mask them out.
        
        # Let's try the simple loop first, if too slow, optimize.
        # Actually, 75k * 56 is 4.2M elements.
        # ~covered[indices] is fast.
        
        # Let's stick to the remaining_indices list but optimize the gain calc.
        # If len(remaining) is large, copy is expensive.
        
        # Better approach:
        # 1. Compute gains for ALL blocks: G = np.sum(~covered[block_t_ids], axis=1)
        # 2. Set G[used] = -1
        # 3. argmax
        # 4. Update covered
        # 5. Add to used
        
        # This computes gains for used blocks too, which is wasted work, but vectorized.
        # 75k * 56 ops = 4.2M ops. Very fast in NumPy.
        
        G = np.sum(~covered[block_t_ids], axis=1)
        G[list(used_blocks)] = -1
        best_idx = np.argmax(G)
        
        if G[best_idx] <= 0:
            # Should not happen if valid
            break
            
        used_blocks.add(best_idx)
        solution.append(list(k_subs[best_idx]))
        
        # Update covered
        covered[block_t_ids[best_idx]] = True
        
    # 4. Local Search: Remove redundant blocks
    # Repeat passes until no change
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        new_solution = []
        # We need to check if a block is redundant.
        # A block is redundant if all its t-subsets are covered by OTHER blocks in the solution.
        
        # Efficient way:
        # 1. Count how many times each t-subset is covered by the current solution.
        # 2. If count[S] > 1 for all S in Block B, then B is redundant.
        
        # Let's build a coverage count array.
        # This is O(num_blocks_in_sol * num_t_in_k)
        # 300 * 56 = 16.8k ops. Very fast.
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in solution:
            # Find index of b in k_subs?
            # We stored k_subs list. We can map block tuple to index.
            pass
            
        # We need a fast way to get block_t_ids for a block in solution.
        # Let's create a dict mapping tuple(block) to index in k_subs.
        block_to_idx = {block: i for i, block in enumerate(k_subs)}
        
        for b in solution:
            idx = block_to_idx[tuple(b)]
            cover_count[block_t_ids[idx]] += 1
            
        # Now check each block in solution
        new_solution = []
        for b in solution:
            idx = block_to_idx[tuple(b)]
            # If min cover count for this block's subsets is > 1, it's redundant
            # Because if it's > 1, removing it leaves count >= 1.
            # If it's 1, it's the only one covering that subset, so keep it.
            min_count = np.min(cover_count[block_t_ids[idx]])
            if min_count > 1:
                changed = True
                # Remove it, update cover_count
                cover_count[block_t_ids[idx]] -= 1
            else:
                new_solution.append(b)
                
        solution = new_solution
        if not changed:
            break

    return solution
