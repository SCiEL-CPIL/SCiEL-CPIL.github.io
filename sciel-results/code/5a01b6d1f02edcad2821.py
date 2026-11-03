import itertools
import random
import time

# Precomputation for the specific case v=19, k=7, t=4
# This allows for very fast greedy and local search steps.

def _precompute_covering_data(v, k, t):
    """
    Precomputes bitmasks for blocks and lists of t-subsets for each block.
    Returns:
    - t_subsets: List of all t-subsets (as tuples)
    - t_to_idx: Dict mapping t-subset tuple to index
    - block_indices: List of all k-block indices (combinations of range(v) of size k)
    - block_masks: List of bitmasks (int) where bit j is set if t_subsets[j] is in the block
    - block_tsub_indices: List of lists of indices of t-subsets contained in the block
    """
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_masks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(block_combinations):
        # Find all t-subsets within this block
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    return t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t

# Global cache for precomputation to avoid re-doing it if called multiple times
# Though construct is called once, it's good practice.
_PRECOMP_CACHE = {}

def construct(v, k, t):
    # Check if we have precomputed data for this specific problem size
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10: # Fallback for larger problems to avoid memory blowup
            # Simple greedy fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            return blocks
            
        _PRECOMP_CACHE[key] = _precompute_covering_data(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t = _PRECOMP_CACHE[key]
    num_blocks = len(block_combinations)
    
    # Number of trials for randomized greedy
    # We use a time budget to ensure we finish within 120s
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_sol = None
    best_len = float('inf')
    
    full_mask = (1 << num_t) - 1
    
    # We will run multiple trials. Each trial consists of:
    # 1. Primal-Dual style Greedy Construction
    # 2. Local Search with Deletions and Swaps
    
    trial_count = 0
    while time.time() - start_time < time_limit:
        trial_count += 1
        random.seed(trial_count)
        uncovered = full_mask
        sol = []
        
        # --- Phase 1: Primal-Dual Approximation Construction ---
        # This simulates the primal-dual algorithm for Set Cover.
        # We maintain dual variables implicitly by tracking uncovered elements.
        # In each step, we increase the "pressure" (dual) on uncovered elements.
        # A block becomes "tight" when the sum of duals of its covered elements
        # reaches its cost (1). Since all costs are 1 and duals increase uniformly,
        # the block with the most uncovered elements becomes tight first.
        # Thus, we select the block covering the maximum number of uncovered t-subsets.
        
        while uncovered:
            best_block_idx = -1
            max_covered = 0
            
            # Iterate through all blocks to find the one covering the most uncovered elements
            for i in range(num_blocks):
                covered = block_masks[i] & uncovered
                if covered:
                    cnt = covered.bit_count()
                    if cnt > max_covered:
                        max_covered = cnt
                        best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            sol.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # --- Phase 2: Local Search (Deletion and Swap) ---
        # We iteratively try to improve the solution.
        
        # Helper function to calculate frequency of t-subset coverage
        def calc_freq(sol_list):
            freq = [0] * num_t
            for b_idx in sol_list:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            return freq
        
        # Helper function to check if a solution is valid (covers all)
        # Since we start from a valid solution and only do safe swaps/deletions,
        # we assume validity is maintained if we check the "lost" elements in swaps.
        
        improved = True
        local_iter = 0
        while improved and local_iter < 100: # Limit local search iterations
            improved = False
            local_iter += 1
            
            # Step 2a: Deletion Pass
            # Remove any block that is redundant
            freq = calc_freq(sol)
            new_sol = []
            for b_idx in sol:
                is_redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_sol.append(b_idx)
            
            if len(new_sol) < len(sol):
                sol = new_sol
                improved = True
                continue # Restart local search after deletion
            
            # Step 2b: Swap Pass
            # Try to swap one block for another to enable further deletions
            # We pick a random block to remove, find the "lost" coverage,
            # and find a block that covers those lost elements best.
            
            if len(sol) > 1:
                # Shuffle to randomize which block we try to remove
                indices_to_try = list(range(len(sol)))
                random.shuffle(indices_to_try)
                
                for idx in indices_to_try:
                    if time.time() - start_time > time_limit:
                        break
                    
                    b_remove = sol[idx]
                    
                    # Calculate coverage of all other blocks
                    # To speed up, we can maintain a "covered by others" mask
                    # But recalculating is safer and simpler for small N
                    others_mask = 0
                    for j, b_idx in enumerate(sol):
                        if j != idx:
                            others_mask |= block_masks[b_idx]
                    
                    # Lost elements: covered by b_remove but NOT by others
                    lost_mask = block_masks[b_remove] & ~others_mask
                    
                    if lost_mask == 0:
                        # Should have been caught by deletion pass, but just in case
                        sol.pop(idx)
                        improved = True
                        break
                    
                    # Find best replacement block
                    # We want a block that covers the most of lost_mask
                    # Ideally, it covers ALL of lost_mask
                    lost_count = lost_mask.bit_count()
                    
                    best_add_idx = -1
                    max_recovered = 0
                    
                    # Iterate through all possible blocks to add
                    # Optimization: Only check blocks that intersect lost_mask
                    for j in range(num_blocks):
                        if j == b_remove:
                            continue
                        # Check if this block is already in sol (by index)
                        # Note: sol contains indices. We need to check if j is in sol.
                        # Since sol is small, `j in sol` is O(N).
                        if j in sol:
                            continue
                        
                        recov = block_masks[j] & lost_mask
                        if recov:
                            cnt = recov.bit_count()
                            if cnt > max_recovered:
                                max_recovered = cnt
                                best_add_idx = j
                                if cnt == lost_count:
                                    break # Perfect replacement found
                    
                    # If we found a block that covers all lost elements, we can swap
                    if best_add_idx != -1 and max_recovered == lost_count:
                        # Perform swap
                        sol[idx] = best_add_idx
                        improved = True
                        # We don't break immediately; we might find more swaps
                        # But to be efficient, let's restart the deletion pass
                        break
                
                if improved:
                    continue
            
            # If no improvement found in this pass, stop local search
            if not improved:
                break
        
        # Update best solution
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            
    if best_sol is None:
        return []
        
    # Convert indices back to lists of points
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks
