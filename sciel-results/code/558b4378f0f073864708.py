"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math


def construct(v, k, t):
    """
    Greedy covering design C(v, k, t) with deadline guard.
    Uses a greedy set-cover heuristic: repeatedly pick the block that covers
    the most uncovered t-subsets. To make this tractable, we build blocks
    greedily by adding the point that maximizes coverage, but we also try
    multiple random restarts and keep the best (fewest blocks) result found
    within the time budget.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubs)}
    
    # We'll track uncovered t-subsets as a boolean array
    # covered[i] = True means tsub i is covered
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts to attempt; adjust based on time
    # For small instances, we can do many restarts
    max_restarts = 50
    restart = 0
    
    while restart < max_restarts:
        if time.time() > deadline - 20:
            break
        
        # Greedy construction with randomization
        uncovered = [True] * num_tsubs
        blocks = []
        uncovered_count = num_tsubs
        
        # Randomize the order in which we consider points for tie-breaking
        point_order = list(range(v))
        random.shuffle(point_order)
        
        while uncovered_count > 0:
            if time.time() > deadline - 5:
                break
            
            # Start with a random t-subset that is still uncovered
            # Find an uncovered t-subset
            start_idx = -1
            for i in range(num_tsubs):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break
            
            # Initialize block with the points of this t-subset
            initial_tsub = all_tsubs[start_idx]
            block = set(initial_tsub)
            
            # Mark all t-subsets contained in this initial t-subset as covered
            # (only the t-subset itself, since it has exactly t elements)
            uncovered[start_idx] = False
            uncovered_count -= 1
            
            # Now greedily add points to maximize coverage
            while len(block) < k:
                if time.time() > deadline - 5:
                    break
                
                best_gain = -1
                best_point = -1
                
                for x in point_order:
                    if x in block:
                        continue
                    
                    # Compute gain: number of uncovered t-subsets that would be covered
                    # if we add x to the block
                    gain = 0
                    # A t-subset is covered by block∪{x} if it's a subset of block∪{x}
                    # and contains x (otherwise it would already be covered by block)
                    # Actually, we need to count t-subsets S such that:
                    #   S ⊆ block ∪ {x}, x ∈ S, and S is not ⊆ block
                    # This is equivalent to: S = T ∪ {x} where T ⊆ block, |T| = t-1
                    # So we count (t-1)-subsets of block that, when unioned with x,
                    # form an uncovered t-subset.
                    
                    for rest in itertools.combinations(sorted(block), t - 1):
                        new_tsub = tuple(sorted(rest + (x,)))
                        if new_tsub in tsub_to_idx:
                            idx = tsub_to_idx[new_tsub]
                            if uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    break
                
                block.add(best_point)
                
                # Update uncovered: mark all t-subsets contained in block as covered
                for tsub in itertools.combinations(sorted(block), t):
                    idx = tsub_to_idx[tsub]
                    if uncovered[idx]:
                        uncovered[idx] = False
                        uncovered_count -= 1
            
            if len(block) >= t:
                blocks.append(sorted(block))
        
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        restart += 1
    
    # If we never found a valid covering, fall back to a simple greedy
    if best_blocks is None:
        # Simple greedy without randomization
        uncovered_set = set(all_tsubs)
        blocks = []
        while uncovered_set:
            block = list(min(uncovered_set))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered_set)
                best_x = max((x for x in range(v) if x not in block), key=gain)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered_set.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
