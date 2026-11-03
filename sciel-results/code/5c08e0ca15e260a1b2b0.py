import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin
    
    # Precompute all t-subsets as bitmasks
    all_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        all_subsets.append(mask)
    
    initial_uncovered = set(all_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(uncovered_set, seed):
        random.seed(seed)
        unc = set(uncovered_set)
        blocks = []
        
        while unc:
            # Pick a random uncovered subset
            target = random.choice(list(unc))
            
            # Extract points from mask
            target_pts = []
            for i in range(v):
                if target & (1 << i):
                    target_pts.append(i)
            
            block = [target_pts[0]]
            block_set = {target_pts[0]}
            
            while len(block) < k:
                best_gain = -1
                candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    if len(block) >= t - 1:
                        # Check all (t-1) subsets of block
                        for combo in itertools.combinations(block, t - 1):
                            mask = (1 << x)
                            for p in combo:
                                mask |= (1 << p)
                            if mask in unc:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain and gain > 0:
                        candidates.append(x)
                
                if not candidates:
                    # No gain found, pick a random point not in block
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        x = random.choice(available)
                    else:
                        break
                else:
                    x = random.choice(candidates)
                
                block.append(x)
                block_set.add(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for combo in itertools.combinations(block, t):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                unc.discard(mask)
                
        return blocks

    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(initial_uncovered, seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        # If we have a good solution, we can keep trying to beat it
        # The greedy is fast, so we can do hundreds of restarts
        
        if time.time() > deadline - 2:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
