"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(v, k, t):
    """
    Covering design C(v, k, t): find as few k-subsets of {0..v-1} as possible
    such that every t-subset is contained in at least one k-subset.
    
    Uses a fast greedy set cover with bitmask-based coverage tracking.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin before the 150s kill
    
    # For small v, we can use bitmasks for fast coverage checks
    # Each t-subset gets a bit in a large integer
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # uncovered_mask: bit i is 1 if t-subset i is still uncovered
    # We'll use a list of booleans for clarity and speed
    uncovered = [True] * num_t
    
    blocks = []
    block_set = set()
    
    def time_check():
        return time.time() >= deadline
    
    # Precompute for each point, which t-subsets contain it
    # point_to_tsubsets[p] = list of t-subset indices that contain point p
    point_to_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            point_to_tsubsets[p].append(i)
    
    # Precompute for each (t-1)-subset, which t-subsets extend it
    # This helps when building a block: if we have a (t-1)-subset S in our block,
    # adding point x covers all t-subsets that are S ∪ {x}
    
    # Greedy construction
    while True:
        if time_check():
            break
        
        # Check if all are covered
        if not any(uncovered):
            break
        
        # Find the first uncovered t-subset
        first_uncovered = -1
        for i in range(num_t):
            if uncovered[i]:
                first_uncovered = i
                break
        
        if first_uncovered == -1:
            break
        
        anchor = all_t_subsets[first_uncovered]
        
        # Build a block starting with the anchor
        block = set(anchor)
        
        # Track which t-subsets are covered by the current block
        # A t-subset is covered if all its points are in the block
        block_set_current = set(block)
        
        def is_covered_by_block(ts_idx, blk):
            ts = all_t_subsets[ts_idx]
            for p in ts:
                if p not in blk:
                    return False
            return True
        
        # Greedily add points
        while len(block) < k:
            if time_check():
                break
            
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Count how many currently uncovered t-subsets would become covered
                gain = 0
                new_block = block | {x}
                for i in range(num_t):
                    if uncovered[i]:
                        if is_covered_by_block(i, new_block):
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                break
            
            block.add(best_point)
        
        block_tuple = tuple(sorted(block))
        
        if block_tuple not in block_set:
            blocks.append(list(block_tuple))
            block_set.add(block_tuple)
        
        # Update uncovered
        for i in range(num_t):
            if uncovered[i]:
                if is_covered_by_block(i, block_tuple):
                    uncovered[i] = False
    
    return blocks
# EVOLVE-BLOCK-END
