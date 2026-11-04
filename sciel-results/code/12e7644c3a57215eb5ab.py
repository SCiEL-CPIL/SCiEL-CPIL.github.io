import itertools
import random
import time

def _remove_redundant_blocks(v, k, t, blocks, counts=None):
    """
    Remove blocks that do not uniquely cover any t-subset.
    If counts is not provided, it is computed.
    Returns a new list of blocks and the updated counts dictionary.
    """
    if not blocks:
        return [], {}
    
    if counts is None:
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
        return blocks, counts
        
    new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
    
    # Update counts for removed blocks
    for i in redundant_indices:
        b = blocks[i]
        for tsub in itertools.combinations(b, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
                
    # Repeat until stable
    while True:
        if not new_blocks:
            return [], counts
            
        redundant_indices = set()
        for i, b in enumerate(new_blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                if counts.get(tsub, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.add(i)
                
        if not redundant_indices:
            break
            
        # Update counts
        for i in redundant_indices:
            b = new_blocks[i]
            for tsub in itertools.combinations(b, t):
                counts[tsub] -= 1
                if counts[tsub] == 0:
                    del counts[tsub]
                    
        new_blocks = [b for i, b in enumerate(new_blocks) if i not in redundant_indices]
        
    return new_blocks, counts

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
            # Precompute t-1 subsets of current block for faster gain calc?
            # block is small (max 7), so itertools is fine.
            block_t1 = list(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in block_t1:
                    # comb is a tuple of t-1 elements.
                    # We need to check if (comb + (x,)) is in uncovered.
                    # Since x is not in block, and block elements are sorted? 
                    # We should keep block sorted or handle tuple sorting.
                    # For speed, let's assume we can just form the tuple and sort if needed.
                    # Actually, `uncovered` contains sorted tuples.
                    # So we must form a sorted tuple.
                    # comb is sorted. x is an int.
                    # We can insert x into comb.
                    # But since v=19, sorting a tuple of 4 is very fast.
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
    # Initialize counts
    counts = {}
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            counts[tsub] = counts.get(tsub, 0) + 1
            
    # Main loop
    while time.time() < deadline:
        if not blocks:
            break
            
        # Pick a random block to remove
        idx = rng.randrange(len(blocks))
        B = blocks[idx]
        
        # Temporarily remove B from counts
        for tsub in itertools.combinations(B, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
                
        # Find uncovered t-subsets (those that were ONLY in B)
        U = []
        for tsub in itertools.combinations(B, t):
            if tsub not in counts: # count was 0
                U.append(tsub)
                
        if not U:
            # B was redundant, removal is valid and improves size
            # Remove B from blocks list
            blocks.pop(idx)
            # Loop continues to remove more redundants
            continue
        
        # We need to cover U with a new block B'
        S_U = set()
        for u in U:
            S_U.update(u)
            
        if len(S_U) > k:
            # Cannot cover U with a single block
            # Restore B
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            continue
        
        # Generate candidate B' by adding random points to S_U
        available_points = [x for x in range(v) if x not in S_U]
        needed = k - len(S_U)
        
        found_better = False
        # Try a few random completions
        for _ in range(10):
            if needed == 0:
                B_prime = sorted(list(S_U))
            else:
                extra = rng.sample(available_points, needed)
                B_prime = sorted(list(S_U) + extra)
            
            # Check if B_prime covers U
            # Since B_prime contains S_U, and U are subsets of S_U, it covers U.
            
            # Tentatively add B_prime to counts
            b_prime_tsubs = list(itertools.combinations(B_prime, t))
            for tsub in b_prime_tsubs:
                counts[tsub] = counts.get(tsub, 0) + 1
            
            # Check if this makes any other block redundant
            # A block is redundant if all its t-subsets have count >= 2.
            # We need to check if any block (other than B_prime which is new) becomes redundant.
            # Actually, B_prime itself might be redundant? No, it covers U which was uncovered, 
            # so it has at least one t-subset with count 1 (the ones in U, now count 1? 
            # Wait. U were count 0. We added B'. So those tsubs are count 1. 
            # So B' is NOT redundant.
            # But other blocks might become redundant because their coverage increased.
            
            redundant_indices = []
            for i, b in enumerate(blocks):
                # We don't check B_prime as it's not in blocks yet.
                # We check existing blocks.
                is_redundant = True
                for tsub in itertools.combinations(b, t):
                    if counts.get(tsub, 0) <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_indices.append(i)
            
            if redundant_indices:
                # Remove redundant blocks from counts
                for i in redundant_indices:
                    b = blocks[i]
                    for tsub in itertools.combinations(b, t):
                        counts[tsub] -= 1
                        if counts[tsub] == 0:
                            del counts[tsub]
                
                # Update blocks list
                # Remove redundant blocks
                # Note: indices in redundant_indices are relative to current blocks list.
                # We also need to remove B (at index idx) and add B_prime.
                
                # It's safer to construct new list
                new_blocks = []
                redundant_set = set(redundant_indices)
                for i, b in enumerate(blocks):
                    if i in redundant_set:
                        continue
                    if i == idx:
                        continue # Skip B
                    new_blocks.append(b)
                
                new_blocks.append(B_prime)
                blocks = new_blocks
                found_better = True
                improved = True
                break # Accept this move and restart loop to clean up
                
            else:
                # No improvement, rollback B_prime counts
                for tsub in b_prime_tsubs:
                    counts[tsub] -= 1
                    if counts[tsub] == 0:
                        del counts[tsub]
                
        if not found_better:
            # Restore B counts
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            # B remains in blocks
            
    # Final cleanup: remove any remaining redundant blocks
    blocks, counts = _remove_redundant_blocks(v, k, t, blocks, counts)
    return blocks

def construct(v, k, t):
    start_time = time.time()
    # Set a safe deadline to ensure we return before 120s
    deadline = start_time + 110.0
    
    all_tsubs = [tuple(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts, but adaptively
    restart_num = 0
    while time.time() < deadline - 10.0:
        restart_num += 1
        rng = random.Random(1000 + restart_num)
        
        # 1. Greedy Construction
        blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline - 15.0)
        
        if not blocks:
            continue
            
        # 2. Local Search
        blocks = _local_search(v, k, t, blocks, rng, deadline - 5.0)
        
        # 3. Final Cleanup
        blocks, _ = _remove_redundant_blocks(v, k, t, blocks)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        # Fallback if no time was left for restarts
        rng = random.Random(0)
        best_blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        best_blocks, _ = _remove_redundant_blocks(v, k, t, best_blocks)
        
    return best_blocks
