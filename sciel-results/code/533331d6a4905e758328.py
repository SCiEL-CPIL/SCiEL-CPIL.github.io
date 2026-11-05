import itertools
import time
import random
import heapq

def _build_inverted_index(block_masks, n_t_subs, n_k_blocks):
    """Build mapping from t-subset index to set of block indices covering it."""
    t_sub_to_blocks = [set() for _ in range(n_t_subs)]
    for bi in range(n_k_blocks):
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_sub_to_blocks[idx].add(bi)
            m ^= lsb
    return t_sub_to_blocks

def _remove_redundant(blocks_idx, block_masks, n_t_subs, deadline):
    """Remove redundant blocks (1-exchange)."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    n = len(current)
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current = []
        for i in range(len(current)):
            bi = current[i]
            mask = block_masks[bi]
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
            if is_critical:
                new_current.append(bi)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
        current = new_current
    
    return current

def _two_exchange(blocks_idx, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline):
    """Try 2-exchanges: remove one block, try to add one different block that covers more."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    current_set = set(current)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(current)
        for i in range(n):
            if time.time() > deadline - 10:
                break
            bi = current[i]
            removed_mask = block_masks[bi]
            
            # Temporarily remove
            new_cover_count = cover_count[:]
            m = removed_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_cover_count[idx] -= 1
                m ^= lsb
            
            # Find uncovered subsets
            uncovered = 0
            for idx, cnt in enumerate(new_cover_count):
                if cnt == 0:
                    uncovered |= (1 << idx)
            
            if uncovered == 0:
                # Can simply remove this block
                new_current = [current[j] for j in range(n) if j != i]
                current = new_current
                current_set = set(current)
                # Rebuild cover_count
                cover_count = [0] * n_t_subs
                for bi2 in current:
                    m = block_masks[bi2]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                improved = True
                break
            else:
                # Try to find a block that covers the most uncovered subsets
                best_gain = 0
                best_block = -1
                # Optimization: only check blocks that cover at least one uncovered subset?
                # Or just iterate all.
                for bi2 in range(n_k_blocks):
                    if bi2 in current_set:
                        continue
                    gain = (block_masks[bi2] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi2
                    if best_gain >= uncovered.bit_count():
                        break
                
                if best_block != -1 and best_gain > 0:
                    if (uncovered & ~block_masks[best_block]) == 0:
                        # Replace bi with best_block
                        new_current = list(current)
                        new_current[i] = best_block
                        current = new_current
                        current_set = set(current)
                        # Rebuild cover_count
                        cover_count = [0] * n_t_subs
                        for bi2 in current:
                            m = block_masks[bi2]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] += 1
                                m ^= lsb
                        improved = True
                        break
    
    return current

def _two_for_one_reduce(blocks_idx, block_masks, t_sub_to_blocks, n_t_subs, deadline):
    """Try to replace 2 blocks with 1 block."""
    if len(blocks_idx) < 2:
        return blocks_idx
    
    current = list(blocks_idx)
    current_set = set(current)
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(current)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            mask_i = block_masks[current[i]]
            for j in range(i + 1, n):
                mask_j = block_masks[current[j]]
                
                # Find t-subsets uncovered by rest if i, j removed
                uncovered_indices = []
                m = mask_i | mask_j
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt = 0
                    if mask_i & lsb: cnt += 1
                    if mask_j & lsb: cnt += 1
                    if cover_count[idx] == cnt:
                        uncovered_indices.append(idx)
                    m ^= lsb
                
                if not uncovered_indices:
                    # Can remove both
                    current = [b for idx, b in enumerate(current) if idx != i and idx != j]
                    current_set = set(current)
                    cover_count = [0] * n_t_subs
                    for bi in current:
                        m = block_masks[bi]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                    improved = True
                    break
                else:
                    # Find candidate blocks
                    # Sort by list size for faster intersection
                    uncovered_indices.sort(key=lambda u: len(t_sub_to_blocks[u]))
                    candidates = set(t_sub_to_blocks[uncovered_indices[0]])
                    for u in uncovered_indices[1:]:
                        if not candidates:
                            break
                        candidates &= t_sub_to_blocks[u]
                    
                    for k in candidates:
                        if k not in current_set:
                            new_current = []
                            for idx, b in enumerate(current):
                                if idx != i and idx != j:
                                    new_current.append(b)
                            new_current.append(k)
                            current = new_current
                            current_set = set(current)
                            cover_count = [0] * n_t_subs
                            for bi in current:
                                m = block_masks[bi]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] += 1
                                    m ^= lsb
                            improved = True
                            break
            if improved:
                break
        
    return current

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Build inverted index for 2-for-1 search
    t_sub_to_blocks = _build_inverted_index(block_masks, n_t_subs, n_k_blocks)
    
    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different greedy orderings
    max_restarts = 5
    restart_time = (deadline - time.time() - 30) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Greedy construction with random tie-breaking
        uncovered_mask = full_mask
        selected = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 30:
                break
            
            best_idx = -1
            best_gain = -1
            best_gain_list = []
            
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_gain_list = [i]
                elif gain == best_gain:
                    best_gain_list.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            # Random tie-breaking for diversity
            if len(best_gain_list) > 1:
                best_idx = random.choice(best_gain_list)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        # Local search loop
        while time.time() < deadline - 10:
            prev_len = len(selected)
            
            # 1. Remove redundant
            selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
            
            # 2. 1-for-1 exchange
            selected = _two_exchange(selected, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline)
            
            # 3. 2-for-1 reduction
            selected = _two_for_one_reduce(selected, block_masks, t_sub_to_blocks, n_t_subs, deadline)
            
            if len(selected) < prev_len:
                continue
            else:
                break
        
        # Verify coverage
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
        
        if time.time() > deadline - 15:
            break
    
    if best_blocks is None:
        # Fallback: greedy without restarts
        uncovered_mask = full_mask
        selected = []
        while uncovered_mask != 0:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
        best_blocks = selected
    
    return [list(all_k_blocks[i]) for i in best_blocks]
