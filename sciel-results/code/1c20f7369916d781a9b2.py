import itertools
import random

def construct(v, k, t):
    all_points = list(range(v))
    
    # Precomputation
    t_subs = list(itertools.combinations(all_points, t))
    n_t = len(t_subs)
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subs)}
    
    k_subs = list(itertools.combinations(all_points, k))
    n_k = len(k_subs)
    
    t_sub_to_blocks = [[] for _ in range(n_t)]
    block_masks = [0] * n_k
    
    for bi, k_sub in enumerate(k_subs):
        k_set = set(k_sub)
        mask = 0
        for ts in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[frozenset(ts)]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(bi)
        block_masks[bi] = mask
        
    full_mask = (1 << n_t) - 1
    
    def solve_instance(seed):
        rng = random.Random(seed)
        
        # Greedy Phase
        uncovered_indices = list(range(n_t))
        uncovered_mask = full_mask
        selected_set = set()
        
        while uncovered_indices:
            idx_t = rng.choice(uncovered_indices)
            candidates = t_sub_to_blocks[idx_t]
            
            best_block = None
            best_gain = -1
            
            for bi in candidates:
                gain = (block_masks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block is None:
                return None
                
            selected_set.add(best_block)
            covered_mask = block_masks[best_block]
            uncovered_mask &= ~covered_mask
            
            # Update uncovered_indices using list comprehension (fast in C)
            uncovered_indices = [i for i in uncovered_indices if (uncovered_mask >> i) & 1]
            
        if uncovered_indices:
            return None
            
        # Local Search Phase
        current_set = set(selected_set)
        current_mask = 0
        for bi in current_set:
            current_mask |= block_masks[bi]
            
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant
            for bi in list(current_set):
                others_mask = current_mask & ~block_masks[bi]
                unique = block_masks[bi] & ~others_mask
                
                if unique == 0:
                    current_set.remove(bi)
                    current_mask &= ~block_masks[bi]
                    improved = True
                    break
            
            # 2. Swap
            if not improved:
                blocks_list = list(current_set)
                rng.shuffle(blocks_list)
                
                for bi in blocks_list:
                    others_mask = current_mask & ~block_masks[bi]
                    unique = block_masks[bi] & ~others_mask
                    
                    if unique == 0:
                        current_set.remove(bi)
                        current_mask &= ~block_masks[bi]
                        improved = True
                        break
                        
                    if unique != 0:
                        # Find replacement
                        u = unique
                        candidates = set()
                        while u:
                            lsb = u & -u
                            idx_t = lsb.bit_length() - 1
                            candidates.update(t_sub_to_blocks[idx_t])
                            u ^= lsb
                        
                        for bj in candidates:
                            if bj not in current_set:
                                if (block_masks[bj] & unique) == unique:
                                    current_set.remove(bi)
                                    current_set.add(bj)
                                    current_mask ^= block_masks[bi]
                                    current_mask |= block_masks[bj]
                                    improved = True
                                    break
                            
                        if improved:
                            break
                            
        if current_mask != full_mask:
            return None
            
        return list(current_set)

    best_solution = None
    best_size = float('inf')
    
    for seed in range(100):
        sol = solve_instance(seed)
        if sol is not None:
            size = len(sol)
            if size < best_size:
                best_size = size
                best_solution = sol
                if best_size <= 264:
                    break
                    
    if best_solution is None:
        raise ValueError("No solution found")
        
    final_blocks = []
    seen = set()
    for bi in best_solution:
        block = list(k_subs[bi])
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            final_blocks.append(block)
            
    return final_blocks
