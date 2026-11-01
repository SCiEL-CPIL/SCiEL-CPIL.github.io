import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 100 # Limit by time too
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        blocks = greedy_construct(v, k, t, all_t_subsets)
        
        # 2. Shrink
        blocks = shrink_covering(blocks, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks if best_blocks is not None else []

def greedy_construct(v, k, t, all_t_subsets):
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        # Pick random uncovered subset
        # random.choice requires a sequence. 
        # Converting set to list is O(N).
        # To avoid O(N) conversion every time, we can keep a list of uncovered indices?
        # Or just accept the cost.
        start_set = random.choice(list(uncovered))
        block = list(start_set)
        
        while len(block) < k:
            # Find best point to add
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain
                # Gain = number of uncovered t-subsets in (block + [x]) that are not in block
                # Actually, just count how many t-subsets of (block+[x]) are in uncovered
                # But we must ensure we don't count ones already covered by previous blocks?
                # 'uncovered' set only contains currently uncovered ones.
                # So any t-subset of (block+[x]) that is in 'uncovered' is a new cover.
                gain = 0
                # Iterate t-subsets of block+[x]
                # block has size k-1. block+[x] has size k.
                # We only care about t-subsets that include x?
                # No, a t-subset of block+[x] might not include x.
                # But if it doesn't include x, it's a t-subset of block.
                # Since we are building the block, the t-subsets of 'block' (size < t? No, block size grows)
                # Wait.
                # If len(block) < t, there are no t-subsets.
                # Here k=8, t=5.
                # Start with size 5.
                # Step 1: block size 5. Add x -> size 6.
                # t-subsets of size 6: 6 choose 5 = 6.
                # One of them is the start_set (which is in uncovered? No, we picked it from uncovered, so it IS in uncovered).
                # So gain is at least 1.
                # We should check all t-subsets of the new block.
                
                new_block = block + [x]
                for sub in itertools.combinations(new_block, t):
                    if sub in uncovered:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen
                break
            block.append(best_point)
            
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            if sub in uncovered:
                uncovered.remove(sub)
                
    return [list(b) for b in blocks]

def shrink_covering(blocks, t):
    if not blocks:
        return blocks
        
    # Count coverage
    cover_count = {}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_count[sub] = cover_count.get(sub, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            # Check if b can be removed
            can_remove = True
            subs_to_check = list(itertools.combinations(b, t))
            for sub in subs_to_check:
                if cover_count[sub] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove b
                for sub in subs_to_check:
                    cover_count[sub] -= 1
                changed = True
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
    return blocks
