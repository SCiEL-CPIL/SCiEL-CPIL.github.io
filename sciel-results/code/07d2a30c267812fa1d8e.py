import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Helper to get indices of t-subsets in a block
    def get_block_t_indices(block):
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # 3. Greedy Initialization
    def greedy_init():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered and time.time() < deadline - 10:
            # Pick a random uncovered t-subset to start the block
            target_idx = random.choice(list(uncovered))
            target_sub = all_t_subsets[target_idx]
            block = list(target_sub)
            
            # Add points to maximize coverage
            while len(block) < k:
                best_point = -1
                best_gain = -1
                
                for p in range(v):
                    if p in block:
                        continue
                    
                    gain = 0
                    if len(block) >= t - 1:
                        for rest in itertools.combinations(block, t - 1):
                            sub = tuple(sorted(rest + (p,)))
                            if sub in t_sub_to_idx:
                                idx = t_sub_to_idx[sub]
                                if idx in uncovered:
                                    gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    candidates = [p for p in range(v) if p not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
            
            if len(block) == k:
                block_tuple = tuple(sorted(block))
                blocks.append(block_tuple)
                for idx in get_block_t_indices(block_tuple):
                    uncovered.discard(idx)
        
        return blocks

    # 4. Simulated Annealing
    blocks = greedy_init()
    
    cover_counts = [0] * num_t_subsets
    for b in blocks:
        for idx in get_block_t_indices(b):
            cover_counts[idx] += 1
            
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    
    best_blocks = list(blocks)
    best_uncovered = uncovered_count
    best_cost = len(blocks) + 100 * uncovered_count
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    block_cache = {}
    def get_indices_cached(block):
        if block not in block_cache:
            block_cache[block] = get_block_t_indices(block)
        return block_cache[block]
        
    def apply_block(block, sign):
        delta_unc = 0
        for idx in get_indices_cached(block):
            old_c = cover_counts[idx]
            new_c = old_c + sign
            cover_counts[idx] = new_c
            if old_c == 0 and new_c == 1:
                delta_unc -= 1
            elif old_c == 1 and new_c == 0:
                delta_unc += 1
        return delta_unc
        
    def random_block():
        return tuple(sorted(random.sample(range(v), k)))
        
    while T > min_T and time.time() < deadline - 5:
        current_cost = len(blocks) + 100 * uncovered_count
        
        r = random.random()
        if r < 0.4:
            # Remove
            if not blocks:
                continue
            idx = random.randrange(len(blocks))
            old_block = blocks[idx]
            
            delta_unc = apply_block(old_block, -1)
            delta_cost = -1 + 100 * delta_unc
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_cost / T)
                
            if accept:
                blocks.pop(idx)
                uncovered_count += delta_unc
                if uncovered_count == 0:
                    new_cost = len(blocks)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(blocks)
                        best_uncovered = 0
            else:
                apply_block(old_block, 1)
                
        elif r < 0.7:
            # Add
            new_block = random_block()
            
            delta_unc = apply_block(new_block, 1)
            delta_cost = 1 + 100 * delta_unc
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_cost / T)
                
            if accept:
                blocks.append(new_block)
                uncovered_count += delta_unc
                if uncovered_count == 0:
                    new_cost = len(blocks)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(blocks)
                        best_uncovered = 0
            else:
                apply_block(new_block, -1)
                
        else:
            # Swap
            if not blocks:
                continue
            idx = random.randrange(len(blocks))
            old_block = blocks[idx]
            new_block = random_block()
            
            if old_block == new_block:
                continue
                
            delta_unc_remove = apply_block(old_block, -1)
            delta_unc_add = apply_block(new_block, 1)
            delta_unc = delta_unc_remove + delta_unc_add
            
            delta_cost = 100 * delta_unc
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_cost / T)
                
            if accept:
                blocks[idx] = new_block
                uncovered_count += delta_unc
                if uncovered_count == 0:
                    new_cost = len(blocks)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(blocks)
                        best_uncovered = 0
            else:
                apply_block(new_block, -1)
                apply_block(old_block, 1)
                
        T *= cooling_rate
        
    # Final Polish: Remove redundant blocks from best_blocks
    if best_uncovered == 0:
        final_blocks = list(best_blocks)
        final_cover_counts = [0] * num_t_subsets
        for b in final_blocks:
            for idx in get_block_t_indices(b):
                final_cover_counts[idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(final_blocks)):
                b = final_blocks[i]
                is_redundant = True
                for idx in get_block_t_indices(b):
                    if final_cover_counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in get_block_t_indices(b):
                        final_cover_counts[idx] -= 1
                    final_blocks.pop(i)
                    changed = True
                    break
                    
        return [list(b) for b in final_blocks]
    else:
        return [list(b) for b in blocks]
