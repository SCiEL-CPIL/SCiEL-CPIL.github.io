import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_index = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)
    
    best_blocks = None
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _greedy_cover(v, k, t, all_tsubsets, tsub_index, deadline, trial)
        
        if blocks is None:
            continue
            
        # Local improvement
        improved_blocks = _local_search(blocks, v, k, t, all_tsubsets, tsub_index, n_tsubs, deadline, trial)
        
        if best_blocks is None or len(improved_blocks) < len(best_blocks):
            best_blocks = improved_blocks
            
        # Early stop if we find a very good solution
        if best_blocks and len(best_blocks) <= 76:
            # We can stop if we hit 76, but maybe we can do better?
            # The lower bound is 69.
            # Let's continue for a bit to see if we can beat 76.
            pass

    if best_blocks is None:
        # Fallback
        blocks = _greedy_cover(v, k, t, all_tsubsets, tsub_index, deadline, -1)
        if blocks:
            best_blocks = _local_search(blocks, v, k, t, all_tsubsets, tsub_index, n_tsubs, deadline, -1)
        else:
            best_blocks = []
            
    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets, tsub_index, deadline, seed_val):
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    uncovered = set(range(len(all_tsubsets)))
    blocks = []
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed_tsub = all_tsubsets[seed_idx]
        block = list(seed_tsub)
        
        while len(block) < k:
            if time.time() > deadline:
                return None
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            best_tiebreakers = []
            
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    idx = tsub_index.get(s)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tiebreakers = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                return None
                
            chosen_x = rng.choice(best_tiebreakers)
            block.append(chosen_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for s in itertools.combinations(block, t):
            idx = tsub_index[s]
            uncovered.discard(idx)
            
    return blocks

def _local_search(blocks, v, k, t, all_tsubsets, tsub_index, n_tsubs, deadline, seed_val):
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    blocks = list(blocks)
    
    # Precompute t-subset indices for each block
    block_tsubs = []
    for b in blocks:
        block_tsubs.append([tsub_index[s] for s in itertools.combinations(b, t)])
        
    # cover_count: array of size n_tsubs
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for tsubs in block_tsubs:
        for idx in tsubs:
            cover_count[idx] += 1
            
    def reduce():
        nonlocal blocks, block_tsubs, cover_count
        changed = True
        while changed:
            if time.time() > deadline:
                return
            changed = False
            # Iterate backwards
            for i in range(len(blocks) - 1, -1, -1):
                redundant = True
                for idx in block_tsubs[i]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    changed = True
    
    reduce()
    
    # Swap loop
    while time.time() < deadline:
        # Check for reduction again
        prev_len = len(blocks)
        reduce()
        if len(blocks) < prev_len:
            continue # Reduction happened, restart swap search
        
        # No reduction possible. Try to find a swap.
        # Find block with minimum number of critical (count==1) t-subsets
        min_crit = float('inf')
        target_idx = -1
        for i in range(len(blocks)):
            crit_count = 0
            for idx in block_tsubs[i]:
                if cover_count[idx] == 1:
                    crit_count += 1
            if crit_count < min_crit:
                min_crit = crit_count
                target_idx = i
                
        if target_idx == -1:
            break
            
        # If min_crit is 0, it should have been reduced. So min_crit >= 1.
        # Try to replace block at target_idx
        old_block = blocks[target_idx]
        old_tsubs = block_tsubs[target_idx]
        
        # Identify critical t-subsets of old_block
        critical_tsubs = [idx for idx in old_tsubs if cover_count[idx] == 1]
        
        # The union of points in these critical t-subsets must be in the new block
        union_points = set()
        for idx in critical_tsubs:
            union_points.update(all_tsubsets[idx])
            
        if len(union_points) > k:
            # Cannot cover all criticals with one block
            # Try next target?
            # To avoid infinite loop, we can shuffle blocks
            rng.shuffle(blocks)
            rng.shuffle(block_tsubs)
            continue
            
        # Generate candidate blocks containing union_points
        base = list(union_points)
        remaining_points = [p for p in range(v) if p not in base]
        
        candidate_found = False
        attempts = 0
        while attempts < 50: # Limit attempts per target
            attempts += 1
            if time.time() > deadline:
                return blocks
                
            # Randomly pick k - len(base) points from remaining
            fillers = rng.sample(remaining_points, k - len(base))
            new_block = tuple(sorted(base + fillers))
            
            # Check if new_block is already in blocks?
            if new_block in blocks:
                continue
                
            # Calculate new t-subsets
            new_tsubs = [tsub_index[s] for s in itertools.combinations(new_block, t)]
            
            # Check validity:
            # All critical t-subsets of old_block must be in new_block
            old_crit_set = set(critical_tsubs)
            new_tsub_set = set(new_tsubs)
            
            if old_crit_set.issubset(new_tsub_set):
                # Valid swap!
                # Perform swap
                for idx in old_tsubs:
                    cover_count[idx] -= 1
                for idx in new_tsubs:
                    cover_count[idx] += 1
                    
                blocks[target_idx] = new_block
                block_tsubs[target_idx] = new_tsubs
                candidate_found = True
                break
                
        if not candidate_found:
            # Shuffle to try different targets
            rng.shuffle(blocks)
            rng.shuffle(block_tsubs)
            
    return blocks
