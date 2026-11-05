import itertools
import random
import sys

def construct(v, k, t):
    points = range(v)
    
    # Enumerate all t-subsets
    t_subs = list(itertools.combinations(points, t))
    n_t = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # Enumerate all k-blocks and compute their coverage as bitmasks
    k_blocks = list(itertools.combinations(points, k))
    n_k = len(k_blocks)
    
    block_masks = [0] * n_k
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << sub_to_idx[t_sub])
        block_masks[i] = mask
    
    # Initialize: greedy seed
    uncovered = (1 << n_t) - 1
    selected = set()
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        selected.add(best_idx)
        uncovered &= ~block_masks[best_idx]
    
    # Local Search: Weighted Max-SAT style
    # Variables: which blocks are selected (binary)
    # Hard constraints: every t-subset must be covered
    # Objective: minimize number of selected blocks (weight = 1 per block)
    
    # We'll use a conflict-directed local search:
    # 1. Start with greedy solution
    # 2. Try to remove blocks (if coverage maintained)
    # 3. Try to swap one block for another if it allows removing more blocks
    # 4. Use dynamic weighting: blocks that cover rare t-subsets get higher priority
    
    # Precompute: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        m = block_masks[i]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            t_to_blocks[bit_idx].append(i)
            m ^= lsb
    
    # Coverage count: how many selected blocks cover each t-subset
    cover_count = [0] * n_t
    for idx in selected:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            cover_count[bit_idx] += 1
            m ^= lsb
    
    best_solution = set(selected)
    best_size = len(selected)
    
    # Local search iterations
    max_iters = 5000
    random.seed(42)
    
    for iteration in range(max_iters):
        # Try to remove a redundant block
        # Find blocks that are not essential (cover_count > 1 for all their t-subsets)
        removable = []
        for idx in selected:
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if cover_count[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                removable.append(idx)
        
        if removable:
            # Remove a random removable block
            idx = random.choice(removable)
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] -= 1
                m ^= lsb
            selected.discard(idx)
            if len(selected) < best_size:
                best_solution = set(selected)
                best_size = len(selected)
            continue
        
        # No removable block: try a swap
        # Pick a block from selected, try replacing it with an unselected block
        # that allows us to remove other blocks later
        
        if not selected:
            break
        
        # Choose a block to potentially replace
        # Prefer blocks that cover t-subsets with high cover_count (more redundancy)
        candidate_blocks = list(selected)
        random.shuffle(candidate_blocks)
        
        swapped = False
        for old_idx in candidate_blocks:
            # Find t-subsets that this block uniquely covers (cover_count == 1)
            unique_ts = []
            m = block_masks[old_idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if cover_count[bit_idx] == 1:
                    unique_ts.append(bit_idx)
                m ^= lsb
            
            # If no unique coverage, this block is redundant (should have been caught above)
            # But due to ordering, it might be essential now. Try to find a replacement.
            
            # Find unselected blocks that cover all unique_ts
            # and have high overlap with other selected blocks
            best_new_idx = -1
            best_score = -1
            
            # Sample some unselected blocks for speed
            unselected = [i for i in range(n_k) if i not in selected]
            if not unselected:
                continue
            
            # Sample up to 100 random unselected blocks
            sample_size = min(100, len(unselected))
            sample = random.sample(unselected, sample_size)
            
            for new_idx in sample:
                # Check if new_idx covers all unique_ts
                m_new = block_masks[new_idx]
                covers_all = True
                for ts in unique_ts:
                    if not (m_new & (1 << ts)):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Score: how many t-subsets does new_idx share with other selected blocks?
                # Higher sharing means more potential for removal
                m_other = 0
                for other_idx in selected:
                    if other_idx != old_idx:
                        m_other |= block_masks[other_idx]
                
                overlap = (m_new & m_other).bit_count()
                # Also consider how many new unique coverages new_idx provides
                # (we want to minimize new unique coverages)
                new_unique = (m_new & ~m_other & ~block_masks[old_idx]).bit_count()
                
                score = overlap - 0.1 * new_unique
                if score > best_score:
                    best_score = score
                    best_new_idx = new_idx
            
            if best_new_idx != -1:
                # Perform the swap
                # Remove old_idx
                m_old = block_masks[old_idx]
                while m_old:
                    lsb = m_old & -m_old
                    bit_idx = lsb.bit_length() - 1
                    cover_count[bit_idx] -= 1
                    m_old ^= lsb
                selected.discard(old_idx)
                
                # Add new_idx
                m_new = block_masks[best_new_idx]
                while m_new:
                    lsb = m_new & -m_new
                    bit_idx = lsb.bit_length() - 1
                    cover_count[bit_idx] += 1
                    m_new ^= lsb
                selected.add(best_new_idx)
                
                swapped = True
                break
        
        if swapped:
            # After swap, try to remove redundant blocks
            for _ in range(10):  # Try a few removals
                removable = []
                for idx in selected:
                    m = block_masks[idx]
                    essential = False
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        if cover_count[bit_idx] == 1:
                            essential = True
                            break
                        m ^= lsb
                    if not essential:
                        removable.append(idx)
                
                if not removable:
                    break
                
                idx = random.choice(removable)
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    cover_count[bit_idx] -= 1
                    m ^= lsb
                selected.discard(idx)
            
            if len(selected) < best_size:
                best_solution = set(selected)
                best_size = len(selected)
        
        # If no improvement in a while, break
        if iteration > 0 and iteration % 500 == 0:
            # Check if we're stuck
            pass
    
    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for idx in list(selected):
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if cover_count[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    cover_count[bit_idx] -= 1
                    m ^= lsb
                selected.discard(idx)
                changed = True
    
    # Use best solution found
    selected = best_solution
    
    # Verify and return
    return [list(k_blocks[i]) for i in sorted(selected)]
