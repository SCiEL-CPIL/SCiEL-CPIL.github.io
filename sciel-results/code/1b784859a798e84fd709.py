import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Helper to get indices of t-subsets of a block
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]

    # 1. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered_mask = [1] * n_t_subs # 1 if uncovered
        
        # To speed up finding uncovered, maintain a list of uncovered indices
        uncovered_indices = list(range(n_t_subs))
        
        while uncovered_indices:
            # Pick an uncovered subset
            # Random pick for diversity
            idx = random.choice(uncovered_indices)
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_p = -1
                best_gain = -1
                # Candidates are points not in block
                candidates = [p for p in range(v) if p not in block]
                
                # Optimization: Precompute which t-subsets of (block + p) contain p?
                # For each p, we look at t-subsets of block U {p} that contain p.
                # These are formed by choosing t-1 elements from block.
                
                for p in candidates:
                    gain = 0
                    # Generate t-subsets of block U {p} containing p
                    # This is equivalent to choosing t-1 elements from block
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx != -1 and uncovered_mask[sub_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        # Tie break randomly?
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1:
                    break # Should not happen
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update counts and uncovered mask
            block_t_subs = get_block_t_sub_indices(block)
            for idx in block_t_subs:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered_mask[idx] = 0
                    # Remove from uncovered_indices?
                    # Removing from list is O(N).
                    # Better: just filter the list occasionally or use a set.
                    # For 11k items, filtering is okay.
            
            # Efficient removal:
            # Since we are iterating, we can just build a new list of uncovered indices
            # But this is O(N) per block.
            # Let's just check `uncovered_mask[idx]` when picking?
            # `random.choice(uncovered_indices)` might pick a covered one.
            # We need to ensure `uncovered_indices` only contains uncovered ones.
            
            # Let's use a set for uncovered indices for O(1) removal.
            # But `random.choice` needs a list.
            # Alternative: Keep a list, and when picking, if it's covered, pick another.
            # Or, just rebuild the list of uncovered indices every M blocks.
            
            # Let's optimize the main loop structure.
            
        return blocks

    # The above greedy is a bit messy with the uncovered list.
    # Let's refine.
    
    def greedy_construct_v2(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered:
            # Pick random uncovered
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    # t-subsets of block U {p} containing p
                    # choose t-1 from block
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1: break
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = t_sub_index[sub]
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks

    # 2. Local Search
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        while time.time() < time_limit:
            # Try to remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b = blocks[i]
                    # Check if all t-subsets of b have count > 1
                    removable = True
                    for sub in itertools.combinations(b, t):
                        if counts[t_sub_index[sub]] <= 1:
                            removable = False
                            break
                    if removable:
                        # Remove block i
                        for sub in itertools.combinations(b, t):
                            counts[t_sub_index[sub]] -= 1
                        blocks.pop(i)
                        changed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break # Restart loop as indices changed
            
            if time.time() >= time_limit:
                break
            
            # Perform random swaps
            # Number of swaps depends on time? Just do a batch.
            for _ in range(100):
                if not blocks: break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in b])
                
                # Check validity
                # Subsets of b containing x
                valid = True
                for sub in itertools.combinations(b, t):
                    if x in sub and counts[t_sub_index[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    # Decrement counts for subsets of b containing x
                    for sub in itertools.combinations(b, t):
                        if x in sub:
                            counts[t_sub_index[sub]] -= 1
                    
                    # Update block
                    b.remove(x)
                    b.append(y)
                    b.sort()
                    blocks[i] = b
                    
                    # Increment counts for subsets of new b containing y
                    for sub in itertools.combinations(b, t):
                        if y in sub:
                            counts[t_sub_index[sub]] += 1
        
        return best_blocks

    # Run multiple greedy constructions and pick the best one for local search
    best_initial = None
    best_initial_len = float('inf')
    
    # Run 5 greedy variants
    for i in range(5):
        if time.time() > deadline - 20: break
        blocks = greedy_construct_v2(i)
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial = blocks
            
    if best_initial is None:
        # Fallback: simple greedy
        best_initial = greedy_construct_v2(0)
        
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
