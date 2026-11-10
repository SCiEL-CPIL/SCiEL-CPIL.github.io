import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    subset_to_blocks = [None] * n_t
    for i in range(n_t):
        subset_to_blocks[i] = []
    for i in range(n_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            subset_to_blocks[idx].append(i)
            m ^= lsb
    
    full_mask = (1 << n_t) - 1
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_idx = -1
            best_gain = -1
            
            for i in range(n_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0 and random.random() < 0.3:
                    best_idx = i
            
            if best_idx == -1:
                break
            
            blocks.append(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return blocks
    
    def prune_redundant(blocks):
        if not blocks:
            return blocks
        
        counts = [0] * n_t
        for b_idx in blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        essential = [False] * len(blocks)
        for i, b_idx in enumerate(blocks):
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    essential[i] = True
                    break
                m ^= lsb
        
        return [blocks[i] for i in range(len(blocks)) if essential[i]]
    
    def find_replacement(unique_mask, exclude_set):
        if unique_mask == 0:
            return None
        
        subs = []
        m = unique_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            subs.append(idx)
            m ^= lsb
        
        if not subs:
            return None
        
        subs_sorted = sorted(subs, key=lambda s: len(subset_to_blocks[s]))
        
        candidates = set(subset_to_blocks[subs_sorted[0]])
        for s in subs_sorted[1:]:
            if len(candidates) == 0:
                break
            candidates &= set(subset_to_blocks[s])
        
        candidates -= exclude_set
        
        for b in candidates:
            return b
        
        return None
    
    def local_search(blocks, max_iters=300):
        if len(blocks) < 2:
            return blocks
        
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline:
            improved = False
            it += 1
            
            cover_mask = 0
            for b_idx in blocks:
                cover_mask |= block_masks[b_idx]
            
            # 2-for-1
            if len(blocks) >= 2:
                for _ in range(40):
                    if time.time() > deadline:
                        return blocks
                    i, j = random.sample(range(len(blocks)), 2)
                    b1, b2 = blocks[i], blocks[j]
                    mask12 = block_masks[b1] | block_masks[b2]
                    others = cover_mask & ~mask12
                    unique = mask12 & ~others
                    
                    if unique == 0:
                        blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                        improved = True
                        break
                    
                    b3 = find_replacement(unique, {b1, b2})
                    if b3 is not None:
                        blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                        blocks.append(b3)
                        improved = True
                        break
            
            # 3-for-2
            if not improved and len(blocks) >= 3:
                for _ in range(15):
                    if time.time() > deadline:
                        return blocks
                    i, j, l = random.sample(range(len(blocks)), 3)
                    b1, b2, b3 = blocks[i], blocks[j], blocks[l]
                    mask123 = block_masks[b1] | block_masks[b2] | block_masks[b3]
                    others = cover_mask & ~mask123
                    unique = mask123 & ~others
                    
                    if unique == 0:
                        blocks = [b for idx, b in enumerate(blocks) if idx not in (i, j, l)]
                        improved = True
                        break
                    
                    # Try 1 replacement
                    b4 = find_replacement(unique, {b1, b2, b3})
                    if b4 is not None:
                        blocks = [b for idx, b in enumerate(blocks) if idx not in (i, j, l)]
                        blocks.append(b4)
                        improved = True
                        break
                    else:
                        # Try 2 replacements
                        subs = []
                        m = unique
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            subs.append(idx)
                            m ^= lsb
                        
                        if len(subs) >= 2:
                            # Split into two groups
                            mid = len(subs) // 2
                            group1 = 0
                            for s in subs[:mid]:
                                group1 |= (1 << s)
                            group2 = 0
                            for s in subs[mid:]:
                                group2 |= (1 << s)
                            
                            b4 = find_replacement(group1, {b1, b2, b3})
                            if b4 is not None:
                                b5 = find_replacement(group2, {b1, b2, b3, b4})
                                if b5 is not None:
                                    blocks = [b for idx, b in enumerate(blocks) if idx not in (i, j, l)]
                                    blocks.append(b4)
                                    blocks.append(b5)
                                    improved = True
                                    break
            
            if improved:
                blocks = prune_redundant(blocks)
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    for i in range(15):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(seed=i * 11 + 3)
        if blocks is None:
            continue
        
        blocks = prune_redundant(blocks)
        blocks = local_search(blocks, max_iters=150)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        blocks = greedy_cover(seed=None)
        best_blocks = blocks if blocks else []
    
    return [list(all_blocks[i]) for i in best_blocks]
