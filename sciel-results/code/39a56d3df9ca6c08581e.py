import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered subset to anchor the block
            target = random.choice(list(uncovered))
            
            best_block = None
            best_gain = -1
            
            # Generate candidates
            # Increase candidates if time allows? Fixed for now.
            num_candidates = 500 
            for _ in range(num_candidates):
                # Generate random block containing target
                remaining = [x for x in range(v) if x not in target]
                needed = k - t
                chosen = list(target) + random.sample(remaining, needed)
                block = tuple(sorted(chosen))
                
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
                
        return blocks

    def local_search(blocks, seed):
        random.seed(seed)
        # Calculate initial coverage counts
        cover_count = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                block = blocks[i]
                # Check if all subsets of this block are covered by other blocks
                redundant = True
                for sub in itertools.combinations(block, t):
                    if cover_count[sub] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                    blocks.pop(i)
                    changed = True
                    # Adjust indices? 
                    # Since we are popping, indices shift. 
                    # It's safer to iterate over a copy of indices or break and restart loop.
                    # Given small N, breaking and restarting is fine.
                    break
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple random restarts
    # How many? 
    # Each run takes ~0.1-0.5s. 
    # We have 110s. 
    # Let's try 50 restarts.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s for last local search
            break
        seed = i
        blocks = greedy_construct(seed)
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback to a single deterministic greedy if something went wrong
        # Or just return empty? No, must be valid.
        # If greedy failed, we have a problem. 
        # But greedy should always succeed.
        pass
        
    return best_blocks
