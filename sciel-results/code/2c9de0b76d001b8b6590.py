import itertools
import random
import time

def _remove_redundant_blocks(v, k, t, blocks):
    """
    Remove blocks that do not uniquely cover any t-subset.
    Returns a new list of blocks.
    """
    if not blocks:
        return []
    
    # Count coverage for each t-subset
    counts = {}
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            counts[tsub] = counts.get(tsub, 0) + 1
            
    redundant_indices = set()
    for i, b in enumerate(blocks):
        is_redundant = True
        for tsub in itertools.combinations(b, t):
            if counts[tsub] <= 1:
                is_redundant = False
                break
        if is_redundant:
            redundant_indices.add(i)
            
    if not redundant_indices:
        return blocks
        
    new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
    
    # Repeat until stable
    while len(new_blocks) < len(blocks):
        blocks = new_blocks
        counts = {}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                counts[tsub] = counts.get(tsub, 0) + 1
                
        redundant_indices = set()
        for i, b in enumerate(blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                if counts[tsub] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.add(i)
                
        if not redundant_indices:
            break
        new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
        
    return new_blocks

def _build_greedy(v, k, t, all_tsubs, rng, deadline):
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = None
            t_minus_1 = t - 1
            block_t1 = set(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in block_t1:
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
                
            block.append(best_point)
            block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tuple(tsub))
            
    return blocks

def _local_search(v, k, t, blocks, rng, deadline):
    # Convert to list of tuples for hashing/comparison if needed, but lists are fine
    # We work with lists of sorted ints
    
    while time.time() < deadline:
        # 1. Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks)
        
        if not blocks:
            break
            
        # 2. Try to improve
        improved = False
        
        # Pick a random block to remove
        idx = rng.randrange(len(blocks))
        B = blocks[idx]
        
        # Temporarily remove B
        blocks.pop(idx)
        
        # Calculate uncovered t-subsets
        # We need to know which t-subsets were ONLY covered by B.
        # Recalculate counts? Or track?
        # Recalculating is O(N * C(k,t)). N~150, C(7,4)=35. 5250 ops. Fast enough.
        
        counts = {}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                counts[tsub] = counts.get(tsub, 0) + 1
        
        # Find t-subsets of B that are now uncovered (count 0)
        U = []
        for tsub in itertools.combinations(B, t):
            if counts.get(tsub, 0) == 0:
                U.append(tsub)
                
        if not U:
            # B was redundant, removal is valid and improves size
            improved = True
            continue # Loop back to remove more redundants
        
        # We need to cover U with a new block B'
        # B' must contain the union of elements in U
        S_U = set()
        for u in U:
            S_U.update(u)
            
        if len(S_U) > k:
            # Cannot cover U with a single block
            # Restore B
            blocks.insert(idx, B)
            # Try a different block? Or just break and try another random one?
            # For now, restore and try next iteration
            continue
        
        # Generate candidate B' by adding random points to S_U
        available_points = [x for x in range(v) if x not in S_U]
        needed = k - len(S_U)
        
        found_better = False
        # Try a few random completions
        for _ in range(5):
            if needed == 0:
                B_prime = sorted(list(S_U))
            else:
                extra = rng.sample(available_points, needed)
                B_prime = sorted(list(S_U) + extra)
            
            # Check if B_prime covers U
            # Since B_prime contains S_U, and U are t-subsets of S_U?
            # Wait. U are t-subsets of B. S_U is union of U.
            # Does B_prime containing S_U imply it covers U?
            # Yes, if u in U, then u is subset of S_U, so u is subset of B_prime.
            
            # Add B_prime
            blocks.append(B_prime)
            
            # Check if this makes any other block redundant
            # Recalculate counts
            counts_new = {}
            for b in blocks:
                for tsub in itertools.combinations(b, t):
                    counts_new[tsub] = counts_new.get(tsub, 0) + 1
            
            # Find redundant blocks in new set
            redundant_indices = []
            for i, b in enumerate(blocks):
                # Skip B_prime (it's the last one, index len-1)
                if i == len(blocks) - 1:
                    continue
                is_redundant = True
                for tsub in itertools.combinations(b, t):
                    if counts_new[tsub] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_indices.append(i)
            
            if redundant_indices:
                # Remove redundant blocks
                # Be careful with indices if we remove multiple
                # We identified indices in the current list.
                # Let's just filter.
                new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
                
                # If we removed at least 1 block, the size decreased by 1 (we added 1, removed 1+1)
                # Original size N.
                # Removed B (N-1).
                # Added B' (N).
                # Removed R redundant (N - len(R)).
                # If len(R) >= 1, size decreases.
                
                blocks = new_blocks
                found_better = True
                improved = True
                break # Accept this move and restart loop to clean up
                
        if not found_better:
            # Restore B
            blocks.pop() # Remove B_prime
            blocks.insert(idx, B)
            # No improvement found with this B. Try next B.
            # To avoid infinite loops on same B, we rely on randomness.
            # But if we always pick the same "bad" B, we waste time.
            # We can shuffle blocks occasionally?
            # For now, just continue.

        # If we improved, the loop continues to clean up.
        # If we didn't, we loop again and pick a new random B.
        
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_tsubs = [tuple(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 20.0: # Leave time for final cleanup
            break
            
        rng = random.Random(1000 + i)
        
        # 1. Greedy Construction
        blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline - 10.0)
        
        # 2. Local Search
        blocks = _local_search(v, k, t, blocks, rng, deadline - 5.0)
        
        # 3. Final Cleanup
        blocks = _remove_redundant_blocks(v, k, t, blocks)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        best_blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        best_blocks = _remove_redundant_blocks(v, k, t, best_blocks)
        
    return best_blocks
