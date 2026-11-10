import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using randomized greedy with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered_initial = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        blocks = randomized_greedy(v, k, t, uncovered_initial, restarts)
        
        # Ensure distinct blocks
        unique_blocks = []
        seen = set()
        for b in blocks:
            tb = tuple(b)
            if tb not in seen:
                seen.add(tb)
                unique_blocks.append(b)
        blocks = unique_blocks
        
        # Simple local search: remove redundant blocks
        blocks = remove_redundant_blocks(blocks, t, v)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Optional: break if we find a very good solution
        if best_count <= 150:
             # We can continue to see if we can do better, but 150 is the known best.
             # Let's run a few more if time permits, but break if time is tight.
             if time.time() > start_time + 100:
                 break

    if best_blocks is None:
        best_blocks = randomized_greedy(v, k, t, uncovered_initial, 0)
        best_blocks = remove_redundant_blocks(best_blocks, t, v)
        
    return best_blocks

def randomized_greedy(v, k, t, initial_uncovered, seed):
    random.seed(seed)
    uncovered = set(initial_uncovered)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        start_subset = random.choice(list(uncovered))
        block = list(start_subset)
        available = [x for x in range(v) if x not in block]
        
        # Add k - t more points
        while len(block) < k:
            if not available:
                break
                
            best_gain = -1
            best_candidates = []
            
            for x in available:
                # Calculate gain
                if len(block) < t - 1:
                    gain = 0
                else:
                    gain = 0
                    # We need to check if adding x covers any uncovered t-subset
                    # The new t-subsets must include x and t-1 points from block
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
                
            x = random.choice(best_candidates)
            block.append(x)
            available.remove(x)
            
        if len(block) < k:
            # Should not happen if v >= k and we started with t points
            # But if v < k, we can't form a block. Problem assumes v >= k.
            break
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for subset in itertools.combinations(block, t):
            uncovered.discard(subset)
            
    return blocks

def remove_redundant_blocks(blocks, t, v):
    """
    Removes blocks that are redundant.
    """
    if not blocks:
        return []
        
    block_subsets = [set(itertools.combinations(b, t)) for b in blocks]
    
    new_blocks = []
    for i, b in enumerate(blocks):
        is_redundant = True
        for s in block_subsets[i]:
            found = False
            for j, other_b in enumerate(blocks):
                if i == j:
                    continue
                if s in block_subsets[j]:
                    found = True
                    break
            if not found:
                is_redundant = False
                break
                
        if not is_redundant:
            new_blocks.append(b)
            
    return new_blocks
