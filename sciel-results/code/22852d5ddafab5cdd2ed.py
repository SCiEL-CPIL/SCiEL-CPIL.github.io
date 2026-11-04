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

def _block_replace(blocks, v, k, t, all_tsubsets_set, deadline):
    """
    Try 1-for-1 block replacement: replace a block with another that covers
    more uncovered t-subsets, potentially allowing subsequent removals.
    """
    if not blocks:
        return blocks
    
    rng = random.Random()
    improved = list(blocks)
    n_blocks = len(improved)
    
    # Build cover count
    cover_count = {}
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] = cover_count.get(s, 0) + 1
    
    # Find which t-subsets are covered exactly once (critical)
    critical = set(s for s, c in cover_count.items() if c == 1)
    
    # Try replacing each block with a better one
    for i in range(n_blocks):
        if time.time() > deadline:
            break
        
        old_block = improved[i]
        old_tsubs = list(itertools.combinations(old_block, t))
        
        # Count how many of old_block's t-subsets are critical
        old_critical_count = sum(1 for s in old_tsubs if s in critical)
        
        if old_critical_count == 0:
            # This block has no critical t-subsets, so it's already redundant
            continue
        
        # Try to find a replacement block that covers all critical t-subsets of old_block
        # and potentially more
        best_new_block = None
        best_gain = -1
        
        # Generate candidate blocks: try random points that include some of old_block
        old_set = set(old_block)
        candidates = set()
        
        # Candidate 1: same block (baseline)
        candidates.add(tuple(sorted(old_block)))
        
        # Generate random candidates by modifying old_block
        for _ in range(50):
            if time.time() > deadline:
                break
            new_block = list(old_block)
            # Remove 1-2 random elements
            remove_count = rng.randint(1, 2)
            to_remove = rng.sample(new_block, remove_count)
            for x in to_remove:
                new_block.remove(x)
            # Add random new elements
            while len(new_block) < k:
                x = rng.randint(0, v-1)
                if x not in new_block:
                    new_block.append(x)
            candidates.add(tuple(sorted(new_block)))
        
        # Evaluate candidates
        for cand in candidates:
            if cand == old_block:
                continue
            
            cand_set = set(cand)
            # Check if cand covers all critical t-subsets of old_block
            covers_all_critical = True
            cand_tsubs = list(itertools.combinations(cand, t))
            
            for s in old_tsubs:
                if s in critical and s not in cand_set:
                    # Need to check if s is actually in cand
                    if tuple(sorted(s)) not in [tuple(sorted(c)) for c in itertools.combinations(cand, t)]:
                        covers_all_critical = False
                        break
            
            if not covers_all_critical:
                continue
            
            # Calculate gain: how many new t-subsets does cand cover that aren't covered by old_block?
            old_tsub_set = set(old_tsubs)
            new_tsub_set = set(cand_tsubs)
            gain = len(new_tsub_set - old_tsub_set)
            
            if gain > best_gain:
                best_gain = gain
                best_new_block = cand
        
        if best_new_block is not None and best_gain > 0:
            # Replace old_block with best_new_block
            # Update cover counts
            for s in old_tsubs:
                cover_count[s] -= 1
            for s in itertools.combinations(best_new_block, t):
                cover_count[s] = cover_count.get(s, 0) + 1
            
            improved[i] = best_new_block
            # Update critical set
            critical = set(s for s, c in cover_count.items() if c == 1)
    
    return improved
