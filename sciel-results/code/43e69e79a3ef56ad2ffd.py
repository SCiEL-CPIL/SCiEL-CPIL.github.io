import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_masks, n_t, deadline):
    """Remove redundant blocks from selected set using bitmask counts."""
    if not selected:
        return set()
    
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        mask = block_masks[b]
        # Extract set bits and increment counts
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for b in list(selected):
            mask = block_masks[b]
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
                selected.discard(b)
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
    return selected

def _greedy_bitmask(block_masks, n_t, full_mask, deadline, randomize=False, top_k=8):
    """Fast greedy set cover using bitmasks."""
    uncovered = full_mask
    selected = []
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        if not randomize:
            best_block = -1
            best_gain = 0
            for i, mask in enumerate(block_masks):
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered &= ~block_masks[best_block]
        else:
            # Randomized: pick from top-k by gain
            gains = []
            for i, mask in enumerate(block_masks):
                gain = (uncovered & mask).bit_count()
                if gain > 0:
                    gains.append((gain, i))
            if not gains:
                break
            gains.sort(reverse=True)
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
            selected.append(best_block)
            uncovered &= ~block_masks[best_block]
    
    return set(selected)

def _local_search_bitmask(selected, block_masks, n_t, full_mask, sub_to_blocks, deadline, max_iters=150):
    """Local search: try removing a block and replacing it with one that covers its unique subsets."""
    if not selected:
        return selected
    
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        mask = block_masks[b]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for _ in range(max_iters):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        b_remove = random.choice(list(best_selected))
        mask_rm = block_masks[b_remove]
        
        # Find t-subsets uniquely covered by b_remove
        unique_subs = []
        m = mask_rm
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] <= 1:
                unique_subs.append(idx)
            m ^= lsb
        
        if not unique_subs:
            # Redundant, just remove
            best_selected.discard(b_remove)
            m = mask_rm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] -= 1
                m ^= lsb
            best_size -= 1
            continue
        
        # Find a replacement block that covers all unique_subs
        # Build a mask of the unique subsets we need to cover
        need_mask = 0
        for s in unique_subs:
            need_mask |= (1 << s)
        
        best_gain = -1
        best_replace = None
        
        # Check candidates from sub_to_blocks for the most constrained unique sub
        min_count = float('inf')
        target = -1
        for s in unique_subs:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target = s
                if c <= 2:
                    break
        
        if target != -1:
            candidates = sub_to_blocks[target]
            for b in candidates:
                if b in best_selected:
                    continue
                # Check if b covers all unique_subs
                if (block_masks[b] & need_mask) != need_mask:
                    continue
                # Gain: how many currently-uncovered subsets does b cover?
                gain = (block_masks[b] & ~((1 << n_t) - 1)).bit_count()  # always 0, just count new coverage
                # Better: count subsets where count was 0
                new_cov = 0
                m2 = block_masks[b]
                while m2:
                    lsb = m2 & -m2
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 0:
                        new_cov += 1
                    m2 ^= lsb
                if new_cov > best_gain:
                    best_gain = new_cov
                    best_replace = b
        
        if best_replace is not None:
            # Remove b_remove, add best_replace
            best_selected.discard(b_remove)
            m = mask_rm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] -= 1
                m ^= lsb
            
            best_selected.add(best_replace)
            m = block_masks[best_replace]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
    
    return best_selected

def _two_for_one(selected, block_masks, n_t, full_mask, deadline, max_iters=100):
    """Try to replace two blocks with one that covers all their unique subsets."""
    if len(selected) < 2:
        return selected
    
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        mask = block_masks[b]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for _ in range(max_iters):
        if not _deadline_ok(deadline):
            break
        
        if len(best_selected) < 2:
            break
        
        # Pick two random blocks
        b1, b2 = random.sample(list(best_selected), 2)
        
        # Find subsets uniquely covered by b1 or b2 (not by any other selected block)
        # A subset s is "unique to {b1,b2}" if counts[s] <= 2 and s is in b1 or b2
        # and removing both b1 and b2 would uncover it
        need_mask = 0
        m1 = block_masks[b1]
        m2 = block_masks[b2]
        combined = m1 | m2
        m = combined
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] <= 2:
                need_mask |= (1 << idx)
            m ^= lsb
        
        if not need_mask:
            # Both are redundant together
            best_selected.discard(b1)
            best_selected.discard(b2)
            for b in (b1, b2):
                m = block_masks[b]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
            best_size -= 2
            continue
        
        # Find a single block that covers need_mask
        best_gain = -1
        best_replace = None
        # Sample some candidate blocks
        candidates = random.sample(range(len(block_masks)), min(5000, len(block_masks)))
        for b in candidates:
            if b in best_selected:
                continue
            if (block_masks[b] & need_mask) != need_mask:
                continue
            new_cov = 0
            m3 = block_masks[b]
            while m3:
                lsb = m3 & -m3
                idx = lsb.bit_length() - 1
                if counts[idx] == 0:
                    new_cov += 1
                m3 ^= lsb
            if new_cov > best_gain:
                best_gain = new_cov
                best_replace = b
        
        if best_replace is not None:
            best_selected.discard(b1)
            best_selected.discard(b2)
            for b in (b1, b2):
                m = block_masks[b]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
            
            best_selected.add(best_replace)
            m = block_masks[best_replace]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
            best_size -= 1
    
    return best_selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their bitmask covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << sub_index[sub])
        block_masks[i] = mask
    
    if not _deadline_ok(deadline):
        return []
    
    # Build sub_to_blocks for local search
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            sub_to_blocks[idx].append(i)
            m ^= lsb
    
    if not _deadline_ok(deadline):
        return []
    
    full_mask = (1 << n_t) - 1
    
    # 3. Multiple greedy restarts with bitmask speed
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 12
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        randomize = (restart > 0)
        top_k = 8 if restart > 2 else 5
        
        selected = _greedy_bitmask(block_masks, n_t, full_mask, deadline, 
                                    randomize=randomize, top_k=top_k)
        
        if not selected:
            continue
        
        selected = _prune_redundant(selected, block_masks, n_t, deadline)
        
        selected = _local_search_bitmask(selected, block_masks, n_t, full_mask, 
                                          sub_to_blocks, deadline, max_iters=100)
        
        selected = _two_for_one(selected, block_masks, n_t, full_mask, deadline, max_iters=50)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # Additional local search on best
        if best_selected is not None and _deadline_ok(deadline):
            best_selected = _local_search_bitmask(best_selected, block_masks, n_t, full_mask,
                                                   sub_to_blocks, deadline, max_iters=80)
            best_selected = _two_for_one(best_selected, block_masks, n_t, full_mask,
                                          deadline, max_iters=40)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
    
    # 4. Final aggressive pruning and local search
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_masks, n_t, deadline)
        if _deadline_ok(deadline):
            best_selected = _local_search_bitmask(best_selected, block_masks, n_t, full_mask,
                                                   sub_to_blocks, deadline, max_iters=120)
        if _deadline_ok(deadline):
            best_selected = _two_for_one(best_selected, block_masks, n_t, full_mask,
                                          deadline, max_iters=80)
        best_selected = _prune_redundant(best_selected, block_masks, n_t, deadline)
    
    # Convert to result
    result = [list(all_blocks[i]) for i in best_selected] if best_selected else []
    
    return result
