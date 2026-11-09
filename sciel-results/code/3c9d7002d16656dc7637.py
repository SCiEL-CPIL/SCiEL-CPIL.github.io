import random

def construct():
    import itertools
    import time
    
    # Hardcoded Witt Design Blocks (132 blocks)
    # I will include a verified set if I can construct it.
    # If I can't, I'll use the randomized greedy.
    
    # For now, let's use the randomized greedy with a time limit.
    # 120s limit.
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Precompute all 6-subsets
    all_blocks = list(itertools.combinations(range(12), 6))
    
    # Precompute 5-subsets for each block for faster checking?
    # Or just use a set of frozensets of 5-subsets.
    
    best_blocks = []
    
    # First, try the deterministic greedy (lexicographic)
    # This is the seed behavior.
    covered = set()
    blocks = []
    for b in all_blocks:
        subsets = list(itertools.combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    best_blocks = blocks
    
    # If we already have 132, we are done.
    if len(best_blocks) == 132:
        return best_blocks
        
    # Try randomized greedy
    for _ in range(1000): # Max restarts
        if time.time() > deadline:
            break
            
        random.shuffle(all_blocks)
        covered = set()
        blocks = []
        for b in all_blocks:
            subsets = list(itertools.combinations(b, 5))
            # Optimization: check if ANY 5-subset is covered
            # We can pre-compute the 5-subsets as a set of frozensets
            # But 'any(s in covered)' is O(6) lookups.
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
                
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            if len(best_blocks) == 132:
                return best_blocks
                
    return best_blocks
