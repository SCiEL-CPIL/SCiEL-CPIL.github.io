import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    subset_set = set(all_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        seed = random.randint(0, 2**31 - 1)
        blocks = greedy_cover(v, k, t, subset_set, seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            
            if time.time() - start_time > time_limit - 10:
                break
            
            # Try local search to improve
            improved = local_search(blocks, v, k, t, time_limit, start_time)
            if improved is not None and len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
            
            # Remove redundant blocks
            blocks = remove_redundant_blocks(best_blocks, v, k, t)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    return best_blocks if best_blocks is not None else []

def greedy_cover(v, k, t, subset_set, seed):
    rng = random.Random(seed)
    uncovered = set(subset_set)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start the block
        # Use a faster method: pick random index from a cached list
        uncovered_list = list(uncovered)
        seed_subset = uncovered_list[rng.randrange(len(uncovered_list))]
        
        block = list(seed_subset)
        rng.shuffle(block)
        
        # Greedily fill the block
        while len(block) < k:
            # Find best point to add using gain computation
            best_gain = -1
            best_point = -1
            candidates = [x for x in range(v) if x not in block]
            rng.shuffle(candidates)
            
            for x in candidates:
                gain = 0
                # Only check (t-1)-subsets of block that could form a t-subset with x
                for rest in itertools.combinations(block, t-1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    if best_gain == 0:
                        break  # Early exit if no gain possible
            
            if best_point == -1 or best_gain <= 0:
                # No point gives positive gain; break
                break
                
            block.append(best_point)
            block.sort()
        
        # Add block to list
        block = tuple(sorted(block))
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
    return blocks

def local_search(blocks, v, k, t, time_limit, start_time):
    """Try to improve by swapping elements in blocks to increase coverage efficiency."""
    if not blocks:
        return blocks
    
    # Build coverage count
    cover_count = {}
    for b in blocks:
        for s in itertools.combinations(b, t):
            cover_count[s] = cover_count.get(s, 0) + 1
    
    improved = list(blocks)
    changed = True
    max_iterations = 50
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit - 5:
            break
        if not changed:
            break
        changed = False
        
        # Try to find a block that can be improved
        for i in range(len(improved)):
            if time.time() - start_time > time_limit - 5:
                break
            b = improved[i]
            # Try removing one element and adding another
            for pos in range(k):
                old_elem = b[pos]
                # Try replacing with each other element
                for new_elem in range(v):
                    if new_elem == old_elem or new_elem in b:
                        continue
                    # Create candidate block
                    new_block = list(b)
                    new_block[pos] = new_elem
                    new_block.sort()
                    new_block = tuple(new_block)
                    
                    # Check if this improves coverage
                    old_subs = set(itertools.combinations(b, t))
                    new_subs = set(itertools.combinations(new_block, t))
                    
                    # Count how many subsets are uniquely covered by old block
                    unique_old = 0
                    for s in old_subs:
                        if cover_count.get(s, 0) == 1:
                            unique_old += 1
                    
                    # Count how many new subsets would be covered
                    new_gain = 0
                    for s in new_subs:
                        if s not in old_subs and cover_count.get(s, 0) == 0:
                            new_gain += 1
                    
                    # Only accept if we don't lose unique coverage and gain new coverage
                    if unique_old == 0 and new_gain > 0:
                        # Accept the change
                        for s in old_subs:
                            cover_count[s] -= 1
                        for s in new_subs:
                            cover_count[s] = cover_count.get(s, 0) + 1
                        improved[i] = new_block
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    
    # Remove redundant blocks after local search
    return remove_redundant_blocks(improved, v, k, t)

def remove_redundant_blocks(blocks, v, k, t):
    if not blocks:
        return blocks
    
    # Count coverage
    cover_count = {}
    for b in blocks:
        for s in itertools.combinations(b, t):
            cover_count[s] = cover_count.get(s, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            # Check if b is redundant
            redundant = True
            for s in itertools.combinations(b, t):
                if cover_count.get(s, 0) < 2:
                    redundant = False
                    break
            
            if redundant:
                # Remove b
                for s in itertools.combinations(b, t):
                    cover_count[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    return blocks
