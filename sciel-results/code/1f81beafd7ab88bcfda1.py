import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # 1. Greedy Initialization
    uncovered_mask = (1 << n_t) - 1
    selected_blocks = []
    
    while uncovered_mask:
        if time.time() > deadline - 30:
            break
        best_block_idx = -1
        best_gain = 0
        for i in range(n_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        if best_block_idx == -1:
            break
        selected_blocks.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # 2. Local Search: Remove Redundant Blocks (1-opt)
    def get_counts(selected):
        counts = [0] * n_t
        for b_idx in selected:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    counts = get_counts(selected_blocks)
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        selected_blocks = new_selected

    best_size = len(selected_blocks)
    best_selected = list(selected_blocks)
    
    # 3. Simulated Annealing / Random Walk on Valid Covers
    current_selected = list(selected_blocks)
    current_counts = get_counts(current_selected)
    current_size = len(current_selected)
    
    T = 1.0
    cooling_rate = 0.95
    
    all_block_indices = list(range(n_blocks))
    
    while time.time() < deadline - 5:
        if T < 0.01:
            break
            
        r = random.random()
        
        if r < 0.3:
            # Try Remove
            if not current_selected: continue
            b_idx = random.choice(current_selected)
            mask = block_masks[b_idx]
            valid = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] <= 1:
                    valid = False
                    break
                m ^= lsb
            if valid:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                current_selected.remove(b_idx)
                new_size = current_size - 1
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_selected = list(current_selected)
        elif r < 0.6:
            # Try Add
            b_idx = random.choice(all_block_indices)
            if b_idx in current_selected: continue
            delta = 1
            if random.random() < pow(2.71828, -delta / T):
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] += 1
                    m ^= lsb
                current_selected.append(b_idx)
                current_size += 1
        else:
            # Try Swap
            if not current_selected: continue
            b_idx = random.choice(current_selected)
            b_idx_new = random.choice(all_block_indices)
            if b_idx == b_idx_new: continue
            if b_idx_new in current_selected: continue
            
            mask_old = block_masks[b_idx]
            mask_new = block_masks[b_idx_new]
            
            critical_subsets = []
            m = mask_old
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] == 1:
                    critical_subsets.append(idx)
                m ^= lsb
                
            if not critical_subsets:
                valid = True
            else:
                valid = True
                for idx in critical_subsets:
                    if not (mask_new & (1 << idx)):
                        valid = False
                        break
            
            if valid:
                m = mask_old
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                m = mask_new
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] += 1
                    m ^= lsb
                
                pos = current_selected.index(b_idx)
                current_selected[pos] = b_idx_new
        
        T *= cooling_rate

    # Final cleanup
    best_selected = list(best_selected)
    counts = get_counts(best_selected)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        new_selected = []
        for b_idx in best_selected:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        best_selected = new_selected

    return [list(all_blocks[i]) for i in best_selected]
