import itertools
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a weighted greedy heuristic
    followed by a deletion pass to remove redundant blocks.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    points = range(v)
    all_t_subs = list(itertools.combinations(points, t))
    all_k_blocks = list(itertools.combinations(points, k))
    
    # Precompute t-subsets for each k-block
    # Map block (tuple) to list of t-subsets (tuples)
    block_tsubs = {}
    for block in all_k_blocks:
        block_tsubs[block] = list(itertools.combinations(block, t))
        
    # Initialize coverage counts
    # Using a dictionary for sparse access, but all t_subs are relevant
    coverage_count = {tsub: 0 for tsub in all_t_subs}
    
    selected_blocks = []
    
    # Greedy Phase
    # To speed up, we can maintain a list of available blocks and remove selected ones?
    # Or just iterate all each time? 50k blocks * ~100-200 iterations = 5-10M operations.
    # Python might be slow. Let's try to optimize the inner loop.
    
    # We will pick the block with max score.
    # Score = sum(1 / (coverage_count[S] + 1) for S in block_tsubs[B])
    
    # To avoid re-calculating scores for all blocks from scratch if possible?
    # It's hard to update scores incrementally easily without a complex data structure.
    # Given the size (50k blocks), a linear scan per iteration is O(N_blocks * t_combinations).
    # 50,000 * 35 = 1.75M operations per iteration.
    # If we need 150 blocks, that's 260M operations. This might be too slow in Python (10-20s+).
    # Let's check if we can prune.
    
    # Alternative: Use a priority queue? No, scores change dynamically.
    # Alternative: Randomized greedy? Pick top K candidates?
    
    # Let's try a simpler greedy first: Maximize number of UNCOVERED subsets.
    # Score = count of S in B where coverage_count[S] == 0.
    # This is faster to compute? Still sum over 35 items.
    
    # Let's implement the weighted version but optimize the loop.
    
    while any(count == 0 for count in coverage_count.values()):
        if time.time() > deadline:
            break
            
        best_block = None
        best_score = -1
        
        # Iterate through all potential blocks
        # Optimization: If a block is already selected, skip it?
        # Yes, blocks must be distinct.
        
        for block in all_k_blocks:
            if block in selected_blocks:
                continue
                
            score = 0.0
            for tsub in block_tsubs[block]:
                # Weighting function
                # Higher weight for uncovered (0), lower for covered
                c = coverage_count[tsub]
                if c == 0:
                    score += 1.0
                else:
                    score += 1.0 / (c + 1.0)
                    
            if score > best_score:
                best_score = score
                best_block = block
                
        if best_block is None:
            break
            
        selected_blocks.append(best_block)
        
        # Update coverage counts
        for tsub in block_tsubs[best_block]:
            coverage_count[tsub] += 1
            
        # Check if all covered
        # Instead of checking all, we can just let the loop condition handle it
        # But 'any(count == 0 ...)' is O(N_tsubs) = 3876. That's fine.
        
    # Deletion Phase
    # Try to remove redundant blocks
    # A block is redundant if all its t-subsets have coverage_count > 1
    # (meaning they are covered by at least one other block)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Iterate backwards to allow removal during iteration
        for i in range(len(selected_blocks) - 1, -1, -1):
            block = selected_blocks[i]
            can_remove = True
            for tsub in block_tsubs[block]:
                if coverage_count[tsub] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                # Remove block
                selected_blocks.pop(i)
                for tsub in block_tsubs[block]:
                    coverage_count[tsub] -= 1
                changed = True
                
    return [list(b) for b in selected_blocks]
