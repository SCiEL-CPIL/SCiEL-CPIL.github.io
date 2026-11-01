import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Precompute subset to index mapping
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Helper to get indices of t-subsets covered by a block
    def get_block_indices(block):
        return [subset_to_idx[subset] for subset in itertools.combinations(block, t)]

    # 2. Greedy Initialization
    def run_greedy():
        uncov = set(range(num_t_subsets))
        blks = []
        while uncov:
            first_idx = min(uncov)
            first_subset = all_t_subsets[first_idx]
            block = list(first_subset)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_subset = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(new_subset)
                        if idx is not None and idx in uncov:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            
            block.sort()
            blks.append(tuple(block))
            
            for idx in get_block_indices(tuple(block)):
                uncov.discard(idx)
        
        return blks

    try:
        initial_blocks = run_greedy()
    except:
        # Fallback to random if greedy fails (unlikely)
        initial_blocks = []

    # Setup SA State
    def calculate_coverage(blks):
        counts = np.zeros(num_t_subsets, dtype=np.int8)
        for b in blks:
            for idx in get_block_indices(b):
                counts[idx] += 1
        return counts

    counts = calculate_coverage(initial_blocks)
    uncovered = int(np.sum(counts == 0))
    num_blocks = len(initial_blocks)
    
    best_blocks = initial_blocks[:]
    best_uncovered = uncovered
    best_num_blocks = num_blocks
    
    # If initial is not valid (shouldn't happen), we must fix it.
    # But greedy always produces valid.
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    def cost_func(uncovered_count, block_count):
        return block_count + 1000 * uncovered_count

    current_cost = cost_func(uncovered, num_blocks)
    
    # Precompute all possible blocks?
    # No, just generate random.
    
    while time.time() < deadline and T > min_T:
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random block
            if num_blocks == 0:
                continue
            idx = random.randint(0, num_blocks - 1)
            block_to_remove = initial_blocks[idx]
            
            for idx2 in get_block_indices(block_to_remove):
                counts[idx2] -= 1
            
            new_uncovered = int(np.sum(counts == 0))
            new_num_blocks = num_blocks - 1
            new_cost = cost_func(new_uncovered, new_num_blocks)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                initial_blocks.pop(idx)
                uncovered = new_uncovered
                num_blocks = new_num_blocks
                current_cost = new_cost
                
                if uncovered == 0 and num_blocks < best_num_blocks:
                    best_num_blocks = num_blocks
                    best_blocks = initial_blocks[:]
            else:
                for idx2 in get_block_indices(block_to_remove):
                    counts[idx2] += 1
        elif move_type < 0.7:
            # Add a random block
            # To make it more useful, maybe add a block that covers many uncovered?
            # But random is fine for SA.
            new_block = tuple(sorted(random.sample(range(v), k)))
            
            for idx2 in get_block_indices(new_block):
                counts[idx2] += 1
            
            new_uncovered = int(np.sum(counts == 0))
            new_num_blocks = num_blocks + 1
            new_cost = cost_func(new_uncovered, new_num_blocks)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                initial_blocks.append(new_block)
                uncovered = new_uncovered
                num_blocks = new_num_blocks
                current_cost = new_cost
            else:
                for idx2 in get_block_indices(new_block):
                    counts[idx2] -= 1
        else:
            # Mutate a block
            if num_blocks == 0:
                continue
            idx = random.randint(0, num_blocks - 1)
            old_block = initial_blocks[idx]
            pos = random.randint(0, k - 1)
            old_point = old_block[pos]
            # Pick a new point different from old_point
            # Avoid duplicates in block
            candidates = [p for p in range(v) if p != old_point and p not in old_block]
            if not candidates:
                continue
            new_point = random.choice(candidates)
            
            new_block_list = list(old_block)
            new_block_list[pos] = new_point
            new_block = tuple(sorted(new_block_list))
            
            if new_block == old_block:
                continue
                
            old_indices = get_block_indices(old_block)
            new_indices = get_block_indices(new_block)
            
            for idx2 in old_indices:
                counts[idx2] -= 1
            for idx2 in new_indices:
                counts[idx2] += 1
                
            new_uncovered = int(np.sum(counts == 0))
            new_num_blocks = num_blocks
            new_cost = cost_func(new_uncovered, new_num_blocks)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                initial_blocks[idx] = new_block
                uncovered = new_uncovered
                current_cost = new_cost
            else:
                for idx2 in old_indices:
                    counts[idx2] += 1
                for idx2 in new_indices:
                    counts[idx2] -= 1
                
        T *= cooling
        
    return [list(b) for b in best_blocks]
