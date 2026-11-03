import itertools
import random

_PRECOMP_CACHE = {}

def _precompute_covering_data(v, k, t):
    """
    Precomputes bitmasks for blocks and lists of t-subsets for each block.
    """
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_masks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(block_combinations):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    return t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t

def construct(v, k, t):
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10:
            # Fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            return blocks
            
        _PRECOMP_CACHE[key] = _precompute_covering_data(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t = _PRECOMP_CACHE[key]
    
    full_mask = (1 << num_t) - 1
    num_blocks = len(block_combinations)
    
    best_sol = None
    best_len = float('inf')
    
    # Primal-Dual / Greedy construction with multiple trials
    # The "Primal-Dual" logic for unit cost set cover is equivalent to:
    # 1. Start with all duals 0.
    # 2. Increase duals of uncovered elements.
    # 3. Select the set that becomes tight first (max coverage of uncovered).
    # This is exactly the Greedy Algorithm.
    
    num_trials = 30
    
    for trial in range(num_trials):
        random.seed(trial * 12345 + 6789)
        uncovered = full_mask
        sol = []
        
        # Greedy Phase (Primal-Dual)
        while uncovered:
            best_block_idx = -1
            max_covered = 0
            
            # To add randomness for tie-breaking, we can iterate in a shuffled order
            # or just pick the first max. Shuffling is expensive if done every step.
            # Instead, we can break ties randomly.
            
            for i in range(num_blocks):
                covered = block_masks[i] & uncovered
                if covered:
                    cnt = covered.bit_count()
                    if cnt > max_covered:
                        max_covered = cnt
                        best_block_idx = i
                    elif cnt == max_covered and cnt > 0:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            sol.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Local Search Phase
        # 1. Remove redundant blocks
        # 2. Try to swap blocks to enable further removals
        
        improved = True
        while improved:
            improved = False
            
            # Calculate coverage frequency
            freq = [0] * num_t
            for b_idx in sol:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            # Identify redundant blocks (those that don't cover any unique t-subset)
            new_sol = []
            for b_idx in sol:
                is_redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_sol.append(b_idx)
                else:
                    improved = True
            
            sol = new_sol
            
            # Recalculate freq for swap logic
            if not sol:
                break
                
            freq = [0] * num_t
            for b_idx in sol:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            # Swap Phase:
            # For each block in sol, check if it can be replaced by another block
            # that covers all the 'critical' t-subsets of the current block.
            # Critical t-subsets are those with freq == 1.
            
            sol_copy = sol[:]
            changed_in_swap = False
            
            for i, b_idx in enumerate(sol_copy):
                # Get critical t-subsets for this block
                criticals = []
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        criticals.append(t_idx)
                
                if not criticals:
                    continue # Redundant, should have been removed, but maybe freq changed?
                    
                # Create a mask of criticals
                critical_mask = 0
                for c in criticals:
                    critical_mask |= (1 << c)
                
                # Find a replacement block b_new not in sol
                # such that block_masks[b_new] covers critical_mask
                found_replacement = False
                # To speed up, we could precompute inverse indices, but let's just scan.
                # Optimization: Only scan blocks not in sol.
                sol_set = set(sol)
                
                for b_new in range(num_blocks):
                    if b_new in sol_set:
                        continue
                    if (block_masks[b_new] & critical_mask) == critical_mask:
                        # Replace
                        sol[i] = b_new
                        found_replacement = True
                        changed_in_swap = True
                        break
                
                # If we made a swap, we need to recalculate freq and restart the loop
                # because the swap might make other blocks redundant or critical.
                if found_replacement:
                    break
            
            if changed_in_swap:
                improved = True
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            
    if best_sol is None:
        return []
        
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks
