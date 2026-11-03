import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 110s out of 120s
    deadline = start_time + 110.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(k_subsets)
    k_subset_to_idx = {s: i for i, s in enumerate(k_subsets)}
    
    masks = [0] * n_k_subsets
    block_indices = [[] for _ in range(n_k_subsets)]
    
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        masks[i] = mask
        block_indices[i] = indices
        
    # 2. Greedy Construction
    uncovered_mask = (1 << n_t_subsets) - 1
    counts = [0] * n_t_subsets
    blocks = []
    
    while uncovered_mask != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in range(n_k_subsets):
            gain = (uncovered_mask & masks[b_idx]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                # Optimization: If we find a block that covers all remaining, we can stop?
                # No, we just need the max gain for this step.
        
        if best_gain == 0:
            break
            
        blocks.append(best_block_idx)
        uncovered_mask &= ~masks[best_block_idx]
        
        for idx in block_indices[best_block_idx]:
            counts[idx] += 1
            
    # 3. Local Search
    def is_redundant(b_idx):
        for idx in block_indices[b_idx]:
            if counts[idx] == 1:
                return False
        return True

    def remove_block(pos):
        b_idx = blocks[pos]
        for idx in block_indices[b_idx]:
            counts[idx] -= 1
        blocks.pop(pos)

    def try_swap(block_pos, p, q):
        b_idx = blocks[block_pos]
        k_sub = k_subsets[b_idx]
        if p not in k_sub or q in k_sub:
            return False
            
        # Check validity
        k_set = set(k_sub)
        for t_sub in itertools.combinations(k_sub, t):
            if p in t_sub:
                idx = t_subset_to_idx[t_sub]
                if counts[idx] == 1:
                    return False
                    
        new_k_sub = tuple(sorted((k_set - {p}) | {q}))
        new_b_idx = k_subset_to_idx[new_k_sub]
        
        # Perform swap
        for idx in block_indices[b_idx]:
            counts[idx] -= 1
        for idx in block_indices[new_b_idx]:
            counts[idx] += 1
            
        blocks[block_pos] = new_b_idx
        
        # Check for redundant blocks
        removed = False
        for i in range(len(blocks)):
            if is_redundant(blocks[i]):
                remove_block(i)
                removed = True
                break
        
        if not removed:
            # Revert
            blocks[block_pos] = b_idx
            for idx in block_indices[new_b_idx]:
                counts[idx] -= 1
            for idx in block_indices[b_idx]:
                counts[idx] += 1
            return False
            
        return True

    while time.time() < deadline:
        # Remove redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if is_redundant(blocks[i]):
                    remove_block(i)
                    changed = True
                    break
        
        if not blocks:
            break
            
        # Local search swaps
        # Limit swaps per loop iteration to allow time check
        for _ in range(50):
            if time.time() > deadline:
                break
            pos = random.randint(0, len(blocks) - 1)
            b_idx = blocks[pos]
            k_sub = k_subsets[b_idx]
            
            p = random.choice(k_sub)
            q_options = [x for x in range(v) if x not in k_sub]
            q = random.choice(q_options)
            
            try_swap(pos, p, q)

    # Final cleanup
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if is_redundant(blocks[i]):
                remove_block(i)
                changed = True
                break

    result = [list(k_subsets[b]) for b in blocks]
    return result
