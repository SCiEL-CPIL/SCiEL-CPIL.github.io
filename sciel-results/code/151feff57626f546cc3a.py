import time
import random
import itertools

# Global cache for precomputed data
_PREPARE_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _PREPARE_CACHE:
        return _PREPARE_CACHE[key]
    
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    n_t = len(t_subs)
    index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Precompute masks for all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    masks = []
    for b in blocks:
        m = 0
        for sub in itertools.combinations(b, t):
            m |= (1 << index[sub])
        masks.append(m)
        
    _PREPARE_CACHE[key] = (blocks, masks, n_t, index)
    return _PREPARE_CACHE[key]

def _greedy_select(blocks, masks, n_t, rng, deadline):
    uncovered = (1 << n_t) - 1
    selected = []
    counts = [0] * n_t
    
    # For speed, we don't need to check all blocks if we can prune, 
    # but 50k is small enough.
    # We will pick the best block each step.
    
    while uncovered and time.time() < deadline:
        best_score = -1
        best_block_idx = -1
        # To allow random tie-breaking, we collect top candidates
        # Or just add random noise to score
        
        # Optimization: 
        # We can skip blocks that cover 0 new items.
        # But we must find the max.
        
        for i, m in enumerate(masks):
            # If this block is already selected, we can't pick it again (blocks are distinct in set cover usually, 
            # and picking same block twice adds no value).
            # Actually, the problem asks for a set of blocks. Duplicates are useless.
            if i in selected:
                continue
            
            new_cov = (m & uncovered).bit_count()
            if new_cov == 0:
                continue
            
            # Score: primarily new coverage. 
            # Tie break: random
            score = new_cov + rng.random() * 1e-6
            
            if score > best_score:
                best_score = score
                best_block_idx = i
        
        if best_block_idx == -1:
            break
            
        selected.append(best_block_idx)
        
        # Update state
        m = masks[best_block_idx]
        uncovered &= ~m
        # Update counts
        # Iterate bits in m
        mm = m
        while mm:
            lb = mm & (-mm)
            idx = lb.bit_length() - 1
            counts[idx] += 1
            mm &= mm - 1
            
    return selected, counts

def _local_search(selected, masks, n_t, deadline, rng):
    # selected is list of indices
    # counts needs to be recomputed or passed. Let's recompute for safety and simplicity in this function.
    counts = [0] * n_t
    for i in selected:
        m = masks[i]
        mm = m
        while mm:
            lb = mm & (-mm)
            idx = lb.bit_length() - 1
            counts[idx] += 1
            mm &= mm - 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try deletions
        to_remove = -1
        for pos, i in enumerate(selected):
            m = masks[i]
            can_remove = True
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    can_remove = False
                    break
                mm &= mm - 1
            if can_remove:
                to_remove = pos
                break
        
        if to_remove != -1:
            i = selected[to_remove]
            m = masks[i]
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                counts[idx] -= 1
                mm &= mm - 1
            selected.pop(to_remove)
            improved = True
            continue
        
        # 2. Try swaps
        # For each block i in selected, find unique coverage
        # Find a block j not in selected that covers unique(i)
        swap_found = False
        for pos, i in enumerate(selected):
            m_i = masks[i]
            unique_mask = 0
            mm = m_i
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    unique_mask |= lb
                mm &= mm - 1
                
            if unique_mask == 0:
                continue # Should have been removed in step 1, but counts might have changed? 
                # No, we just checked. If unique is 0, it's removable.
                # Wait, if unique is 0, it means all its bits have count > 1.
                # So it SHOULD be removable. Why didn't step 1 catch it?
                # Step 1 checks `counts[idx] == 1`. If none are 1, it's removable.
                # So if we reach here, unique_mask must be non-zero.
                
            # Find j not in selected such that masks[j] covers unique_mask
            # To speed up, we can just scan.
            for j in range(len(masks)):
                if j in selected:
                    continue
                if (masks[j] & unique_mask) == unique_mask:
                    # Swap i with j
                    # Remove i
                    mm = m_i
                    while mm:
                        lb = mm & (-mm)
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        mm &= mm - 1
                    # Add j
                    m_j = masks[j]
                    mm = m_j
                    while mm:
                        lb = mm & (-mm)
                        idx = lb.bit_length() - 1
                        counts[idx] += 1
                        mm &= mm - 1
                    
                    selected[pos] = j
                    swap_found = True
                    improved = True
                    break
            if swap_found:
                break
                
    return selected

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0 # Safety margin
    
    blocks, masks, n_t, index = _prepare(v, k, t)
    n_blocks = len(blocks)
    
    best_size = float('inf')
    best_selected = []
    
    # Multiple restarts
    # Each restart: Greedy + Local Search
    # Time per restart: Greedy ~0.5s, LS ~1-2s. 
    # We can do ~10-20 restarts.
    
    num_restarts = 20
    time_per_restart = (deadline - t0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        rng = random.Random(42 + r)
        
        # Greedy
        sel, counts = _greedy_select(blocks, masks, n_t, rng, deadline - 2.0)
        
        # Local Search
        sel = _local_search(sel, masks, n_t, deadline - 2.0, rng)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = sel
            
    # Final validity check and fallback
    # Ensure all t-subsets are covered
    covered = 0
    for i in best_selected:
        covered |= masks[i]
        
    if covered != (1 << n_t) - 1:
        # Fallback: add any missing blocks greedily
        uncovered = ((1 << n_t) - 1) & ~covered
        for i in range(n_blocks):
            if i in best_selected:
                continue
            if masks[i] & uncovered:
                best_selected.append(i)
                uncovered &= ~masks[i]
                if not uncovered:
                    break

    # Return list of blocks (as lists of ints)
    result = [list(blocks[i]) for i in best_selected]
    return result
