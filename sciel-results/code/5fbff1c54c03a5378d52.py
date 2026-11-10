"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Greedy construction with vectorized state
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []
    
    while not covered.all():
        if time.time() > deadline - 30:  # Reserve time for local search
            break
        
        # Find first uncovered subset
        uncovered_mask = ~covered
        if not uncovered_mask.any():
            break
        start_idx = np.where(uncovered_mask)[0][0]
        start_sub = all_subs[start_idx]
        
        # Build block greedily starting from start_sub
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        idx = sub_to_idx[candidate]
                        if not covered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True
    
    if not covered.all():
        # Fallback: if greedy didn't finish, use pure greedy without time limit
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    # Phase 2: Redundancy removal using conflict graph concept
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 10:
        coverage = np.zeros(n_subs, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                coverage[idx] += 1
        
        new_blocks = []
        for block in blocks:
            has_unique = False
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                if coverage[idx] == 1:
                    has_unique = True
                    break
            
            if not has_unique:
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx[sub]
                    coverage[idx] -= 1
            else:
                new_blocks.append(block)
        
        blocks = new_blocks
    
    # Phase 3: Simulated annealing local search
    if time.time() < deadline - 5:
        blocks = _local_search(blocks, v, k, t, sub_to_idx, all_subs, deadline)
    
    return blocks


def _local_search(blocks, v, k, t, sub_to_idx, all_subs, deadline):
    """Simulated annealing with swap and remove operations."""
    rng = random.Random(42)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    n_subs = len(all_subs)
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    temp = 2.0
    cooling = 0.997
    min_temp = 0.05
    
    while time.time() < deadline - 2 and temp > min_temp:
        move_type = rng.randint(0, 2)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a random block
            idx = rng.randint(0, len(current_blocks))
            block = current_blocks[idx]
            
            can_remove = True
            for sub in itertools.combinations(block, t):
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for sub in itertools.combinations(block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                current_blocks.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
            else:
                temp *= cooling
        
        elif move_type == 1 and len(current_blocks) > 0:
            # Try to swap a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            
            # Find a replacement block that covers more uncovered subsets
            best_new_block = None
            best_delta = 0
            
            for _ in range(20):
                candidate = sorted(rng.sample(range(v), k))
                
                # Compute delta: how many new subsets covered vs how many lost
                new_covered = 0
                for sub in itertools.combinations(candidate, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        new_covered += 1
                
                lost = 0
                for sub in itertools.combinations(old_block, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 1:
                        lost += 1
                
                delta = new_covered - lost
                if delta > best_delta:
                    best_delta = delta
                    best_new_block = candidate
            
            if best_new_block is not None and (best_delta > 0 or rng.random() < np.exp(best_delta / temp)):
                # Apply swap
                for sub in itertools.combinations(old_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in itertools.combinations(best_new_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = best_new_block
                
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(current_blocks)
                
                temp *= cooling
            else:
                temp *= cooling
        
        else:
            # Try to add a block that covers many uncovered subsets (if we're not optimal yet)
            uncovered_mask = coverage == 0
            if uncovered_mask.any() and len(current_blocks) < 160:
                best_block = None
                best_gain = -1
                
                for _ in range(15):
                    candidate = sorted(rng.sample(range(v), k))
                    gain = 0
                    for sub in itertools.combinations(candidate, t):
                        sidx = sub_to_idx[sub]
                        if coverage[sidx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate
                
                if best_block is not None and best_gain > 0:
                    for sub in itertools.combinations(best_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] += 1
                    current_blocks.append(best_block)
                    
                    if len(current_blocks) < best_size:
                        best_blocks = list(current_blocks)
                        best_size = len(current_blocks)
                    
                    temp *= cooling
                else:
                    temp *= cooling
            else:
                temp *= cooling
    
    return best_blocks
# EVOLVE-BLOCK-END
