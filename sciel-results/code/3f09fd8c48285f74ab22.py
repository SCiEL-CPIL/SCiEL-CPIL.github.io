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

    # 1. Maximal t-packing: find blocks with disjoint t-subsets to establish structure
    def maximal_t_packing(seed, time_limit):
        random.seed(seed)
        # Track which t-subsets are "used" by the packing
        used_t_subs = [0] * n_t_subs
        packing_blocks = []
        
        while time.time() < time_limit:
            # Find a block that maximizes uncovered t-subsets
            best_block = None
            best_gain = -1
            
            # Try random starting points
            for attempt in range(50):
                if time.time() >= time_limit:
                    break
                
                # Start with a random t-subset
                start_idx = random.randint(0, n_t_subs - 1)
                block = list(all_t_subs[start_idx])
                
                # Greedily extend to k elements
                while len(block) < k:
                    best_p = -1
                    best_p_gain = -1
                    candidates = [p for p in range(v) if p not in block]
                    
                    for p in candidates:
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            sub = tuple(sorted(rest + (p,)))
                            idx = t_sub_index.get(sub, -1)
                            if idx != -1 and used_t_subs[idx] == 0:
                                gain += 1
                        if gain > best_p_gain:
                            best_p_gain = gain
                            best_p = p
                        elif gain == best_p_gain and best_p != -1 and random.random() < 0.3:
                            best_p = p
                    
                    if best_p == -1:
                        break
                    block.append(best_p)
                
                block.sort()
                
                # Calculate total gain for this block
                total_gain = 0
                for sub in itertools.combinations(block, t):
                    idx = t_sub_index[sub]
                    if used_t_subs[idx] == 0:
                        total_gain += 1
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_block = block[:]
            
            if best_block is None or best_gain <= 0:
                break
            
            # Add to packing
            packing_blocks.append(best_block)
            for sub in itertools.combinations(best_block, t):
                idx = t_sub_index[sub]
                used_t_subs[idx] += 1
        
        return packing_blocks, used_t_subs

    # 2. Greedy construction guided by packing scaffold
    def greedy_construct_guided(seed, packing_blocks, packing_used, time_limit):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        # First, add packing blocks (they provide structural coverage)
        for pb in packing_blocks:
            blocks.append(pb)
            for sub in itertools.combinations(pb, t):
                idx = t_sub_index[sub]
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
        
        # Now greedily fill remaining gaps
        while uncovered and time.time() < time_limit:
            # Prioritize t-subsets that are "isolated" (fewer covering options)
            # Pick a random uncovered subset
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
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
                
                if best_p == -1:
                    break
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx2 = t_sub_index[sub]
                counts[idx2] += 1
                if counts[idx2] == 1:
                    uncovered.discard(idx2)
                    
        return blocks

    # 3. Local Search
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

    # Strategy: Use packing-guided greedy with multiple seeds
    best_initial = None
    best_initial_len = float('inf')
    
    # Time budget allocation
    packing_time = min(10, (deadline - start_time) * 0.1)
    greedy_time = min(30, (deadline - start_time) * 0.3)
    search_time = max(20, deadline - start_time - packing_time - greedy_time - 5)
    
    # Try packing-guided approach
    packing_start = time.time()
    packing_blocks, packing_used = maximal_t_packing(42, packing_start + packing_time)
    
    # Run guided greedy with different seeds
    for i in range(3):
        if time.time() > deadline - search_time - 5:
            break
        blocks = greedy_construct_guided(i, packing_blocks, packing_used, time.time() + greedy_time/3)
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial = blocks
    
    # Also try standard greedy for comparison
    if time.time() < deadline - search_time - 10:
        # Simple greedy without packing
        def greedy_construct_simple(seed):
            random.seed(seed)
            counts = [0] * n_t_subs
            blocks = []
            uncovered = set(range(n_t_subs))
            
            while uncovered:
                idx = random.choice(list(uncovered))
                target_sub = all_t_subs[idx]
                block = list(target_sub)
                
                while len(block) < k:
                    best_p = -1
                    best_gain = -1
                    candidates = [p for p in range(v) if p not in block]
                    
                    for p in candidates:
                        gain = 0
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
                
                for sub in itertools.combinations(block, t):
                    idx2 = t_sub_index[sub]
                    counts[idx2] += 1
                    if counts[idx2] == 1:
                        uncovered.discard(idx2)
                        
            return blocks
        
        simple_blocks = greedy_construct_simple(0)
        if len(simple_blocks) < best_initial_len:
            best_initial_len = len(simple_blocks)
            best_initial = simple_blocks
    
    if best_initial is None:
        best_initial = greedy_construct_simple(0)
    
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
