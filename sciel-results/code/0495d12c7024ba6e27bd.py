"""Improved covering design solver using fast greedy and local search."""
import time
import random
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Precompute bitmask for each t-subset for fast subset checking
    tsub_masks = [0] * num_tsubsets
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsub_masks[i] = mask
    
    # For each point, precompute list of (tsub_idx, mask_without_point)
    # This allows fast gain computation: gain(x) = count of uncovered tsubs where 
    # tsub_mask & ~block_mask == (1 << x) ... actually we need tsub contains x and rest in block
    # Better: for point x, store list of (tsub_idx, mask_without_x) where mask_without_x = tsub_mask & ~(1<<x)
    point_tsub_info = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        mask = tsub_masks[i]
        for p in ts:
            point_tsub_info[p].append((i, mask & ~(1 << p)))
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = bytearray(num_tsubsets)
        for i in range(num_tsubsets):
            uncovered[i] = 1
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None
            
            # Pick an uncovered t-subset to seed the block
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            block_set = set(block)
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block_set]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for ts_idx, mask_without_x in point_tsub_info[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        # Check if mask_without_x is subset of block_mask
                        if (mask_without_x & ~block_mask) == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                        best_point = x
                
                if best_point == -1:
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
                block_mask |= (1 << best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered using bitmask
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                if (tsub_masks[ts_idx] & ~block_mask) == 0:
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    def local_search(blocks, seed, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        rng = random.Random(seed)
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Precompute coverage lists for each block
        block_covered = []
        for b in blocks:
            b_mask = 0
            for p in b:
                b_mask |= (1 << p)
            covered_list = []
            for ts_idx in range(num_tsubsets):
                if (tsub_masks[ts_idx] & ~b_mask) == 0:
                    covered_list.append(ts_idx)
            block_covered.append(covered_list)
        
        # Count coverage
        cover_count = [0] * num_tsubsets
        for cl in block_covered:
            for ts_idx in cl:
                cover_count[ts_idx] += 1
        
        kept = [True] * n
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for bi in order:
                if time.time() > time_limit:
                    break
                if not kept[bi]:
                    continue
                
                # Check if block bi is redundant
                redundant = True
                for ts_idx in block_covered[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    kept[bi] = False
                    improved = True
                    for ts_idx in block_covered[bi]:
                        cover_count[ts_idx] -= 1
        
        new_blocks = [b for i, b in enumerate(blocks) if kept[i]]
        
        # Try swap-based improvement: replace one block with a better one
        if time.time() < time_limit - 5:
            current_count = len(new_blocks)
            for _ in range(50):  # limited number of swap attempts
                if time.time() > time_limit - 2:
                    break
                if len(new_blocks) <= current_count:
                    break
                
                # Pick a random block to try replacing
                bi = rng.randrange(len(new_blocks))
                b_mask = 0
                for p in new_blocks[bi]:
                    b_mask |= (1 << p)
                
                # Find t-subsets uniquely covered by this block
                unique_tsubs = []
                for ts_idx in block_covered[bi]:
                    if kept[bi] and cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                
                if not unique_tsubs:
                    # Block is redundant, just remove it
                    kept[bi] = False
                    for ts_idx in block_covered[bi]:
                        cover_count[ts_idx] -= 1
                    new_blocks.pop(bi)
                    current_count -= 1
                    # Recompute block_covered for remaining blocks
                    block_covered = []
                    for b in new_blocks:
                        bm = 0
                        for p in b:
                            bm |= (1 << p)
                        cl = []
                        for ts_idx in range(num_tsubsets):
                            if (tsub_masks[ts_idx] & ~bm) == 0:
                                cl.append(ts_idx)
                        block_covered.append(cl)
                    continue
                
                # Try to find a new block that covers all unique_tsubs
                # This is a set cover subproblem - use greedy
                new_block_candidates = set()
                for ts_idx in unique_tsubs:
                    ts = all_tsubsets[ts_idx]
                    for combo in itertools.combinations(range(v), k - len(ts)):
                        candidate = tuple(sorted(list(ts) + list(combo)))
                        new_block_candidates.add(candidate)
                
                if len(new_block_candidates) > 200:
                    # Too many candidates, skip
                    continue
                
                best_new_block = None
                best_gain = -1
                for candidate in new_block_candidates:
                    cand_mask = 0
                    for p in candidate:
                        cand_mask |= (1 << p)
                    # Check if it covers all unique_tsubs
                    covers_all = True
                    for ts_idx in unique_tsubs:
                        if (tsub_masks[ts_idx] & ~cand_mask) != 0:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Count how many additional t-subsets it covers
                    add_gain = 0
                    for ts_idx in range(num_tsubsets):
                        if cover_count[ts_idx] == 0 and (tsub_masks[ts_idx] & ~cand_mask) == 0:
                            add_gain += 1
                    if add_gain > best_gain:
                        best_gain = add_gain
                        best_new_block = candidate
                
                if best_new_block is not None:
                    # Replace the block
                    old_mask = 0
                    for p in new_blocks[bi]:
                        old_mask |= (1 << p)
                    # Remove old block's coverage
                    for ts_idx in block_covered[bi]:
                        cover_count[ts_idx] -= 1
                    # Add new block's coverage
                    new_mask = 0
                    for p in best_new_block:
                        new_mask |= (1 << p)
                    new_covered = []
                    for ts_idx in range(num_tsubsets):
                        if (tsub_masks[ts_idx] & ~new_mask) == 0:
                            cover_count[ts_idx] += 1
                            new_covered.append(ts_idx)
                    new_blocks[bi] = list(best_new_block)
                    block_covered[bi] = new_covered
        
        return new_blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        blocks = local_search(blocks, 0, time_limit=deadline - 15)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 25:
        blocks = greedy_cover(seed)
        if blocks is not None:
            blocks = local_search(blocks, seed, time_limit=deadline - 5)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        return []
    
    return best_blocks
