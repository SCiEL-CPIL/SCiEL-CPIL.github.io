import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, trial)
        if valid and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
        
        # Local improvement: try to remove redundant blocks
        if valid:
            improved_blocks = _local_improve(blocks, v, k, t, deadline)
            if len(improved_blocks) < len(blocks):
                blocks = improved_blocks
                if best_blocks is None or len(blocks) < len(best_blocks):
                    best_blocks = blocks
            
            # Local search with swaps
            if time.time() < deadline:
                swapped_blocks = _swap_local_search(blocks, v, k, t, deadline)
                if len(swapped_blocks) < len(blocks):
                    blocks = swapped_blocks
                    if best_blocks is None or len(blocks) < len(best_blocks):
                        best_blocks = blocks
        
        # If we find a very good solution, we might stop early?
        # The known best is 76. If we hit 76, we can stop.
        if best_blocks and len(best_blocks) <= 76:
            break
            
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, -1)
        if valid:
            best_blocks = blocks
            # Try to improve even the fallback
            improved_blocks = _local_improve(best_blocks, v, k, t, deadline)
            if len(improved_blocks) < len(best_blocks):
                best_blocks = improved_blocks
            if time.time() < deadline:
                swapped_blocks = _swap_local_search(best_blocks, v, k, t, deadline)
                if len(swapped_blocks) < len(best_blocks):
                    best_blocks = swapped_blocks
        else:
            # Last resort: return empty list (invalid, but avoids crash)
            best_blocks = []
            
    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy covering with randomized tie-breaking.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = set(all_tsubsets_set)
    blocks = []
    
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered t-subset as seed
        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            best_tiebreakers = []
            
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tiebreakers = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                return None, False
                
            chosen_x = rng.choice(best_tiebreakers)
            block.append(chosen_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks, True

def _local_improve(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    """
    if not blocks:
        return blocks
    
    # Build a count for each t-subset: how many blocks cover it
    cover_count = {}
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] = cover_count.get(s, 0) + 1
    
    n_blocks = len(blocks)
    improved = list(blocks)
    
    # Try removing each block if it's redundant
    # We iterate backwards so that removal doesn't affect indices
    for i in range(n_blocks - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Check if block i is redundant: all its t-subsets are covered by other blocks
        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break
        
        if redundant:
            # Remove block i
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)
    
    return improved

def _swap_local_search(blocks, v, k, t, deadline):
    """
    Local search using 1-swap moves.
    Tries to swap a point in a block with a point outside the block,
    provided that no triple becomes uncovered (i.e., no triple with count 1 is lost).
    After each successful swap, checks for and removes redundant blocks.
    """
    if not blocks:
        return blocks
    
    improved = [list(b) for b in blocks]
    
    # Build cover count
    cover_count = {}
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] = cover_count.get(s, 0) + 1
            
    points = list(range(v))
    
    while time.time() < deadline:
        any_change = False
        
        # Iterate through blocks
        for i in range(len(improved)):
            if time.time() > deadline:
                return improved
                
            block_i = improved[i]
            block_i_set = set(block_i)
            old_tsubs = set(itertools.combinations(block_i, t))
            
            # Try swapping each point p in block_i with q not in block_i
            for p in block_i:
                for q in points:
                    if q in block_i_set:
                        continue
                    
                    new_block = [x for x in block_i if x != p]
                    new_block.append(q)
                    new_block.sort()
                    new_tsubs = set(itertools.combinations(new_block, t))
                    
                    lost_tsubs = old_tsubs - new_tsubs
                    
                    # Check if any lost triple is critical (count 1)
                    valid = True
                    for s in lost_tsubs:
                        if cover_count[s] == 1:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                    
                    # Swap is valid
                    # Update cover counts
                    for s in lost_tsubs:
                        cover_count[s] -= 1
                    added_tsubs = new_tsubs - old_tsubs
                    for s in added_tsubs:
                        cover_count[s] = cover_count.get(s, 0) + 1
                        
                    improved[i] = new_block
                    any_change = True
                    break # Break q loop
                
                if any_change:
                    break # Break p loop
                    
            if any_change:
                # Restart block loop? We break out of i loop to restart scan
                break
        
        if any_change:
            # Try to remove redundant blocks
            improved = _local_improve(improved, v, k, t, deadline)
            # Rebuild cover_count
            cover_count = {}
            for block in improved:
                for s in itertools.combinations(block, t):
                    cover_count[s] = cover_count.get(s, 0) + 1
        else:
            # No more swaps found
            break
            
    return improved
