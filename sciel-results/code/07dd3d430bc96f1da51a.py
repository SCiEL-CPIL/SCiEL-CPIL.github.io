import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Check for bit_count
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    # Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    block_list = []
    mask_list = []
    block_to_t_sub_indices = []
    t_sub_to_blocks = {sub: [] for sub in t_subs}
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        sub_indices = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
            sub_indices.append(idx)
            t_sub_to_blocks[sub].append(len(block_list))
        
        block_list.append(block)
        mask_list.append(mask)
        block_to_t_sub_indices.append(sub_indices)
        
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline - 5.0:
        covered_mask = 0
        coverage_count = [0] * num_t_subs
        used_blocks = []
        used_block_indices = []
        used_set = set()
        
        uncovered = list(range(num_t_subs))
        
        while uncovered:
            idx = random.choice(uncovered)
            sub = t_subs[idx]
            
            candidates = t_sub_to_blocks[sub]
            
            best_gain = -1
            best_b_idx = -1
            
            for b_idx in candidates:
                if b_idx in used_set:
                    continue
                
                mask = mask_list[b_idx]
                gain = popcount(mask & ~covered_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            if best_b_idx == -1:
                break
                
            b = block_list[best_b_idx]
            used_blocks.append(b)
            used_block_indices.append(best_b_idx)
            used_set.add(best_b_idx)
            
            mask = mask_list[best_b_idx]
            covered_mask |= mask
            
            for sub_idx in block_to_t_sub_indices[best_b_idx]:
                coverage_count[sub_idx] += 1
                
            new_uncovered = [u for u in uncovered if not (mask >> u) & 1]
            uncovered = new_uncovered
            
        # Local Search: Remove Redundant Blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(used_blocks)):
                b_idx = used_block_indices[i]
                is_redundant = True
                for sub_idx in block_to_t_sub_indices[b_idx]:
                    if coverage_count[sub_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub_idx in block_to_t_sub_indices[b_idx]:
                        coverage_count[sub_idx] -= 1
                    
                    used_blocks.pop(i)
                    used_block_indices.pop(i)
                    used_set.remove(b_idx)
                    changed = True
                    break
        
        if len(used_blocks) < best_len:
            best_len = len(used_blocks)
            best_blocks = [list(b) for b in used_blocks]
            
    if best_blocks is None:
        return []
        
    return best_blocks
