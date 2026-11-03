import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Helper: Greedy Set Cover
    def greedy_cover(uncovered_mask, used_blocks_set):
        selected = []
        uncovered = uncovered_mask
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(n_blocks):
                if i in used_blocks_set:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            selected.append(best_idx)
            used_blocks_set.add(best_idx)
            uncovered &= ~masks[best_idx]
        return selected

    # Helper: Remove Redundant Blocks
    def remove_redundant(blocks_list):
        if not blocks_list:
            return []
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                # Check if blocks_list[i] is redundant
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                if union_others == all_mask:
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    # Main Logic: Iterated Local Search
    start_time = time.time()
    best_solution = []
    best_size = float('inf')
    
    # 1. Initial Greedy
    used = set()
    initial = greedy_cover(all_mask, used)
    initial = remove_redundant(initial)
    best_solution = initial[:]
    best_size = len(initial)
    
    # 2. ILS Loop
    max_time = 110.0
    
    while time.time() - start_time < max_time:
        if not best_solution:
            break
            
        current = best_solution[:]
        # Perturb: Remove 1-3 random blocks
        num_remove = min(3, len(current))
        for _ in range(num_remove):
            if current:
                idx = random.randint(0, len(current) - 1)
                current.pop(idx)
        
        # Repair: Greedily add blocks to cover gaps
        uncovered = all_mask
        used_set = set(current)
        for b in current:
            uncovered &= ~masks[b]
        
        additions = greedy_cover(uncovered, used_set)
        current = current + additions
        
        # Local Optimize
        current = remove_redundant(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_solution = current[:]
            
    return [list(k_subsets[i]) for i in best_solution]
