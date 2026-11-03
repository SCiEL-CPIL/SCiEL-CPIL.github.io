import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    points = list(range(v))
    
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
    
    # 4. Greedy Set Cover
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
    
    best_solution = selected_greedy[:]
    current_best = len(best_solution)
    
    # 5. Fast redundancy removal using bitset union
    def remove_redundant(blocks_list):
        if not blocks_list:
            return blocks_list
        n = len(blocks_list)
        # Compute prefix and suffix unions for O(1) check if block i is redundant
        # prefix[i] = union of masks[blocks_list[0..i-1]]
        # suffix[i] = union of masks[blocks_list[i+1..n-1]]
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] | masks[blocks_list[i]]
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] | masks[blocks_list[i]]
        
        removed = True
        while removed and blocks_list:
            removed = False
            n = len(blocks_list)
            prefix = [0] * (n + 1)
            suffix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] | masks[blocks_list[i]]
            for i in range(n-1, -1, -1):
                suffix[i] = suffix[i+1] | masks[blocks_list[i]]
            for i in range(n):
                union_others = prefix[i] | suffix[i+1]
                if union_others == all_mask:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list
    
    best_solution = remove_redundant(best_solution)
    current_best = len(best_solution)
    
    # 6. Local search: try to remove blocks by finding alternative coverage
    def local_search(blocks_list, max_iters=50):
        blocks_list = blocks_list[:]
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < 110:
            improved = False
            iters += 1
            n = len(blocks_list)
            if n == 0:
                break
            
            # Compute coverage count for each t-subset
            # For each block, determine which t-subsets it uniquely covers
            # A block is redundant if no t-subset is uniquely covered by it
            
            # Efficient approach: for each block, compute the t-subsets it covers
            # that are NOT covered by any other block in the list
            
            # Build a list of masks for current blocks
            block_masks = [masks[b] for b in blocks_list]
            
            # For each block i, compute unique = block_masks[i] & ~(union of all other blocks)
            # Use prefix/suffix to compute union of others in O(1) per block
            prefix = [0] * (n + 1)
            suffix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] | block_masks[i]
            for i in range(n-1, -1, -1):
                suffix[i] = suffix[i+1] | block_masks[i]
            
            for i in range(n):
                union_others = prefix[i] | suffix[i+1]
                unique = block_masks[i] & ~union_others
                if unique == 0:
                    # Block i is redundant
                    blocks_list.pop(i)
                    improved = True
                    break
        
        return blocks_list
    
    best_solution = local_search(best_solution)
    current_best = len(best_solution)
    
    # 7. Multi-start greedy with different orderings
    for restart in range(20):
        if time.time() - start_time > 100:
            break
        
        # Different greedy strategies
        if restart == 0:
            # Standard greedy
            uncovered = all_mask
            selected = []
            used = [False] * n_blocks
            while uncovered:
                best_gain = -1
                best_idx = -1
                for i in range(n_blocks):
                    if used[i]:
                        continue
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected.append(best_idx)
                used[best_idx] = True
                uncovered &= ~masks[best_idx]
        elif restart < 10:
            # Greedy with random tie-breaking
            uncovered = all_mask
            selected = []
            used = [False] * n_blocks
            while uncovered:
                best_gain = -1
                best_candidates = []
                for i in range(n_blocks):
                    if used[i]:
                        continue
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
                if not best_candidates:
                    break
                idx = random.choice(best_candidates)
                selected.append(idx)
                used[idx] = True
                uncovered &= ~masks[idx]
        else:
            # Greedy with penalty for blocks that cover many already-covered t-subsets
            uncovered = all_mask
            selected = []
            used = [False] * n_blocks
            while uncovered:
                best_score = -1
                best_idx = -1
                for i in range(n_blocks):
                    if used[i]:
                        continue
                    gain = (masks[i] & uncovered).bit_count()
                    if gain == 0:
                        continue
                    # Score: prefer blocks with high gain, penalize overlap with already selected
                    # Simple: just use gain
                    score = gain
                    if score > best_score:
                        best_score = score
                        best_idx = i
                if best_idx == -1:
                    break
                selected.append(best_idx)
                used[best_idx] = True
                uncovered &= ~masks[best_idx]
        
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iters=30)
        
        if len(selected) < current_best:
            current_best = len(selected)
            best_solution = selected[:]
    
    return [list(k_subsets[i]) for i in best_solution]
