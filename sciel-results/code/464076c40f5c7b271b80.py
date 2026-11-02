import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            i = t_index_map[combo]
            mask |= (1 << i)
        block_masks[idx] = mask

    full_mask = (1 << num_t) - 1

    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            candidates = []
            for i in range(num_k):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = rng.choice(candidates)
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    def remove_redundant(selected_indices):
        if not selected_indices:
            return []
        
        cover_count = [0] * num_t
        for idx in selected_indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                cover_count[idx_t] += 1
                m ^= lsb
        
        kept = []
        for idx in selected_indices:
            mask = block_masks[idx]
            is_unique = False
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                if cover_count[idx_t] == 1:
                    is_unique = True
                    break
                m ^= lsb
            
            if is_unique:
                kept.append(idx)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    cover_count[idx_t] -= 1
                    m ^= lsb
        
        return kept

    def local_search(selected_indices, max_iter=50):
        current = list(selected_indices)
        if not current:
            return current
        
        cover_count = [0] * num_t
        for idx in current:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                cover_count[idx_t] += 1
                m ^= lsb
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            n = len(current)
            if n < 2:
                break
            
            # 1-for-0
            for i in range(n):
                mask = block_masks[current[i]]
                is_unique = False
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if cover_count[idx_t] == 1:
                        is_unique = True
                        break
                    m ^= lsb
                
                if not is_unique:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        cover_count[idx_t] -= 1
                        m ^= lsb
                    current.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2-for-1
            # Try to find i, j such that unique(i) U unique(j) is covered by some k
            # We'll sample pairs to keep it fast
            rng = random.Random(iter_count)
            tried = 0
            while tried < 1000 and time.time() < deadline - 5.0:
                tried += 1
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                    
                unique_i = 0
                unique_j = 0
                mask_i = block_masks[current[i]]
                mask_j = block_masks[current[j]]
                
                m = mask_i
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if cover_count[idx_t] == 1:
                        unique_i |= lsb
                    m ^= lsb
                    
                m = mask_j
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if cover_count[idx_t] == 1:
                        unique_j |= lsb
                    m ^= lsb
                    
                union_unique = unique_i | unique_j
                if union_unique == 0:
                    continue
                    
                # Find k that covers union_unique
                # k must not be i or j
                # We'll check a random subset of blocks
                for _ in range(10):
                    k_idx = rng.randint(0, num_k - 1)
                    if k_idx == current[i] or k_idx == current[j]:
                        continue
                    if (union_unique & ~block_masks[k_idx]) == 0:
                        # Found a swap
                        # Remove i, j, add k_idx
                        # Update cover counts
                        m = mask_i
                        while m:
                            lsb = m & -m
                            idx_t = lsb.bit_length() - 1
                            cover_count[idx_t] -= 1
                            m ^= lsb
                            
                        m = mask_j
                        while m:
                            lsb = m & -m
                            idx_t = lsb.bit_length() - 1
                            cover_count[idx_t] -= 1
                            m ^= lsb
                            
                        mask_k = block_masks[k_idx]
                        m = mask_k
                        while m:
                            lsb = m & -m
                            idx_t = lsb.bit_length() - 1
                            cover_count[idx_t] += 1
                            m ^= lsb
                            
                        # Update current list
                        # Remove j first (assuming j > i? no, just remove by value)
                        # List remove is O(N).
                        current.remove(current[j])
                        current.remove(current[i])
                        current.append(k_idx)
                        
                        improved = True
                        break
                        
                if improved:
                    break
                    
        return current

    best_blocks = None
    best_len = float('inf')

    attempts = 0
    while time.time() < deadline - 10.0:
        seed = attempts
        selected = greedy_init(seed)
        if selected is None:
            continue
        
        selected = remove_redundant(selected)
        selected = local_search(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
            
        if best_len <= 330:
            break
            
        attempts += 1
        
    if best_blocks is None:
        selected = greedy_init(0)
        if selected:
            selected = remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
