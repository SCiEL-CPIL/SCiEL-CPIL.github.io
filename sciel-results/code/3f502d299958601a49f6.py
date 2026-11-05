import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_masks = [0] * num_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << num_t_subsets) - 1
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_construct():
        uncovered = full_mask
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            best_gain = 0
            best_block_idx = -1
            for b_idx in range(num_blocks):
                inter = uncovered & block_masks[b_idx]
                if inter:
                    gain = popcount(inter)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            if best_block_idx == -1:
                break
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        return selected_blocks

    def reduce_blocks(block_indices):
        covered = 0
        for b_idx in block_indices:
            covered |= block_masks[b_idx]
            
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                if (covered & ~block_masks[b_idx]) == full_mask:
                    covered &= ~block_masks[b_idx]
                    changed = True
                else:
                    new_block_indices.append(b_idx)
            block_indices = new_block_indices
        return block_indices

    # 1. Greedy
    sol = greedy_construct()
    sol = reduce_blocks(sol)
    best_solution = sol
    best_size = len(sol)
    
    # 2. Local Search
    current_solution = best_solution
    current_covered = 0
    for b_idx in current_solution:
        current_covered |= block_masks[b_idx]
    sol_set = set(current_solution)
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    max_swaps = 10000
    swaps_attempted = 0
    
    while time.time() < deadline - 5 and swaps_attempted < max_swaps:
        if not current_solution:
            break
            
        idx_in_sol = random.randint(0, len(current_solution) - 1)
        old_block = current_solution[idx_in_sol]
        
        new_block = random.randint(0, num_blocks - 1)
        if new_block in sol_set:
            continue
            
        new_covered = (current_covered & ~block_masks[old_block]) | block_masks[new_block]
        
        if new_covered != full_mask:
            continue
            
        current_solution[idx_in_sol] = new_block
        sol_set.remove(old_block)
        sol_set.add(new_block)
        current_covered = new_covered
        swaps_attempted += 1
        
        new_sol = reduce_blocks(current_solution)
        if len(new_sol) < len(current_solution):
            current_solution = new_sol
            sol_set = set(current_solution)
            current_covered = 0
            for b_idx in current_solution:
                current_covered |= block_masks[b_idx]
            
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = current_solution.copy()
                
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
