import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute for each k-subset, which t-subsets it covers (as a set of indices)
    # This is expensive to do for all C(v,k) subsets, so we'll do it on-demand or for a pool
    
    # Strategy: Greedy set cover with a large candidate pool, followed by local search (block swaps)
    # to reduce the number of blocks.
    
    best_blocks = None
    best_size = float('inf')
    
    def compute_block_t_indices(block):
        """Return set of t-subset indices covered by this block."""
        return set(t_sub_to_idx[sub] for sub in itertools.combinations(block, t))
    
    def greedy_cover(pool, time_limit):
        """
        Greedy set cover using the given pool of candidates.
        pool: list of (block_tuple, set_of_t_indices)
        Returns (selected_blocks, is_valid, time_exceeded)
        """
        uncovered = set(range(n_t))
        selected = []
        candidates = pool[:]
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            best_block = None
            best_tset = None
            
            # Find block with max gain
            for i, (bt, bsubs) in enumerate(candidates):
                gain = len(uncovered & bsubs)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_block = bt
                    best_tset = bsubs
            
            if best_gain == 0:
                # Stuck: need to generate a new block on the fly
                # Pick the uncovered t-subset with smallest index
                target_idx = min(uncovered)
                target = all_t_subsets[target_idx]
                block = list(target)
                # Add random points to reach size k
                while len(block) < k:
                    cand = random.randint(0, v-1)
                    if cand not in block:
                        block.append(cand)
                block.sort()
                bt = tuple(block)
                bsubs = compute_block_t_indices(bt)
                selected.append(bt)
                uncovered -= bsubs
                # Don't add to candidates (one-off)
                if time.time() > time_limit:
                    return selected, False, True
                continue
            
            selected.append(best_block)
            uncovered -= best_tset
            candidates.pop(best_idx)
            
            if time.time() > time_limit:
                return selected, False, True
        
        return selected, True, False
    
    def remove_redundancies(blocks):
        """Remove blocks that are not essential (no unique t-subset coverage)."""
        if not blocks:
            return blocks
        
        # Count coverage
        coverage_count = {}
        for b in blocks:
            for s in itertools.combinations(b, t):
                idx = t_sub_to_idx[s]
                coverage_count[idx] = coverage_count.get(idx, 0) + 1
        
        new_blocks = []
        for b in blocks:
            essential = False
            for s in itertools.combinations(b, t):
                if coverage_count[t_sub_to_idx[s]] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(b)
        
        # Iterate until stable
        changed = True
        while changed:
            changed = False
            coverage_count = {}
            for b in new_blocks:
                for s in itertools.combinations(b, t):
                    idx = t_sub_to_idx[s]
                    coverage_count[idx] = coverage_count.get(idx, 0) + 1
            
            next_blocks = []
            for b in new_blocks:
                essential = False
                for s in itertools.combinations(b, t):
                    if coverage_count[t_sub_to_idx[s]] == 1:
                        essential = True
                        break
                if essential:
                    next_blocks.append(b)
                else:
                    changed = True
            
            new_blocks = next_blocks
        
        return new_blocks
    
    def local_search(blocks, time_limit):
        """
        Try to reduce block count by swapping blocks.
        For each block, try removing it and adding a different block that covers the lost t-subsets.
        Also try direct swaps.
        """
        blocks = list(blocks)
        n_blocks = len(blocks)
        
        # Precompute t-subset coverage for current blocks
        # For each t-subset index, which blocks cover it?
        t_to_blocks = {}
        for i, b in enumerate(blocks):
            for s in itertools.combinations(b, t):
                idx = t_sub_to_idx[s]
                if idx not in t_to_blocks:
                    t_to_blocks[idx] = []
                t_to_blocks[idx].append(i)
        
        # Try to remove blocks that have all their t-subsets covered by others
        # This is similar to remove_redundancies but we can be more aggressive
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Recompute coverage
            t_to_blocks = {}
            for i, b in enumerate(blocks):
                for s in itertools.combinations(b, t):
                    idx = t_sub_to_idx[s]
                    if idx not in t_to_blocks:
                        t_to_blocks[idx] = []
                    t_to_blocks[idx].append(i)
            
            # Try to remove each block
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    break
                b = blocks[i]
                # Check if all t-subsets of b are covered by other blocks
                can_remove = True
                for s in itertools.combinations(b, t):
                    idx = t_sub_to_idx[s]
                    # Is there another block covering this?
                    covering = [j for j in t_to_blocks.get(idx, []) if j != i]
                    if not covering:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(i)
                    improved = True
                    break  # Restart loop after removal
        
        return blocks
    
    # Generate initial pool of random blocks
    pool_size = 5000
    pool = []
    seen = set()
    while len(pool) < pool_size:
        b = tuple(sorted(random.sample(range(v), k)))
        if b not in seen:
            seen.add(b)
            bsubs = compute_block_t_indices(b)
            pool.append((b, bsubs))
    
    # Run greedy cover
    selected, valid, exceeded = greedy_cover(pool, deadline - 20)
    
    if valid:
        selected = remove_redundancies(selected)
        best_blocks = selected
        best_size = len(selected)
    
    # Try local search to improve
    if best_blocks is not None and time.time() < deadline - 10:
        improved = local_search(best_blocks, deadline - 5)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Try multiple restarts if time permits
    restarts = 0
    while time.time() < deadline - 30:
        restarts += 1
        # Generate new pool
        pool = []
        seen = set()
        while len(pool) < pool_size:
            b = tuple(sorted(random.sample(range(v), k)))
            if b not in seen:
                seen.add(b)
                bsubs = compute_block_t_indices(b)
                pool.append((b, bsubs))
        
        selected, valid, exceeded = greedy_cover(pool, deadline - 25)
        if valid:
            selected = remove_redundancies(selected)
            if len(selected) < best_size:
                best_blocks = selected
                best_size = len(selected)
        
        if time.time() < deadline - 10:
            improved = local_search(best_blocks, deadline - 5)
            if len(improved) < best_size:
                best_blocks = improved
                best_size = len(improved)
    
    if best_blocks is None:
        # Fallback: simple greedy without pool
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            target_idx = min(uncovered)
            target = all_t_subsets[target_idx]
            block = list(target)
            while len(block) < k:
                # Greedy point selection
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x not in block:
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            sub = tuple(sorted(rest + (x,)))
                            if t_sub_to_idx.get(sub, -1) in uncovered:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_x = x
                if best_x == -1:
                    best_x = random.randint(0, v-1)
                    while best_x in block:
                        best_x = random.randint(0, v-1)
                block.append(best_x)
            block.sort()
            bt = tuple(block)
            blocks.append(bt)
            uncovered -= compute_block_t_indices(bt)
        best_blocks = blocks
    
    return [list(b) for b in best_blocks]
