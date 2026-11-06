import random
import time

N = 256
MAX_SUM = 2 * N  # 512

def _bitmask_add_check(sel_mask, sums_mask, x):
    """Check if x can be added to a Sidon set represented by bitmasks."""
    # 2*x must not be in sums
    if sums_mask >> (2 * x) & 1:
        return False
    # For each a in sel, x+a must not be in sums
    # We need to check: for all a in sel, (x+a) not in sums
    # Equivalent: (sums_mask shifted left by x) & sel_mask == 0
    # Because if a is in sel and x+a is in sums, then bit a of sel AND bit (x+a) of sums
    # After shifting sums left by x: bit (a-x) of shifted sums corresponds to bit a of sums
    # Wait, let me think again.
    # We want: for all a in sel, (x + a) not in sums
    # sums_mask has bit s set if s is a sum.
    # We need: no a in sel such that (x+a) in sums
    # i.e., sel_mask & (sums_mask >> x) == 0  ... no that's wrong direction
    # If sums_mask has bit (x+a) set, then (sums_mask >> x) has bit a set.
    # So we need: sel_mask & (sums_mask >> x) == 0
    if sel_mask & (sums_mask >> x):
        return False
    return True

def _update_masks(sel_mask, sums_mask, x):
    """Add x to the set and update sum masks."""
    new_sel = sel_mask | (1 << x)
    # Add all sums x + a for a in sel (including x itself)
    # For each a in sel, add sum x+a
    # sums_mask needs to have bits x+a set for all a in sel, plus 2*x
    # We can compute: for each a where sel_mask has bit a, set bit (x+a) in sums
    # This is: (sel_mask << x) gives bits at positions x+a for each a in sel
    # But we also need 2*x which is x+x, and since x is being added, x is in new_sel
    # Actually x is NOT yet in sel_mask, so we need to add 2*x separately
    # Or: new_sel includes x, so (new_sel << x) would include 2*x but also x+old_a
    # Let's just do: new_sums = sums_mask | (sel_mask << x) | (1 << (2*x))
    # But wait, we also need sums among the old elements which are already in sums_mask
    new_sums = sums_mask | (sel_mask << x) | (1 << (2 * x))
    return new_sel, new_sums

def _is_maximal(sel_mask, sums_mask):
    """Check if no element in [1,N] can be added to the Sidon set."""
    for x in range(1, N + 1):
        if sel_mask >> x & 1:
            continue
        if _bitmask_add_check(sel_mask, sums_mask, x):
            return False
    return True

def _count_bits(mask):
    return bin(mask).count('1')

def _local_search_shrink(sel_mask, sums_mask, deadline, max_iters=500):
    """Try to remove elements and re-saturate to get a smaller maximal Sidon set."""
    best_mask = sel_mask
    best_sums = sums_mask
    best_size = _count_bits(sel_mask)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try removing each element
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not (sel_mask >> x & 1):
                continue
            
            # Remove x
            cand_mask = sel_mask & ~(1 << x)
            # Recompute sums from scratch for the candidate
            cand_sums = 0
            elems = [a for a in range(1, N + 1) if cand_mask >> a & 1]
            for i in range(len(elems)):
                for j in range(i, len(elems)):
                    cand_sums |= 1 << (elems[i] + elems[j])
            
            # Saturate: add all addable elements
            changed = True
            while changed:
                changed = False
                for y in range(1, N + 1):
                    if cand_mask >> y & 1:
                        continue
                    if _bitmask_add_check(cand_mask, cand_sums, y):
                        cand_mask, cand_sums = _update_masks(cand_mask, cand_sums, y)
                        changed = True
            
            if _count_bits(cand_mask) < best_size and _is_maximal(cand_mask, cand_sums):
                best_mask = cand_mask
                best_sums = cand_sums
                best_size = _count_bits(cand_mask)
                sel_mask = cand_mask
                sums_mask = cand_sums
    
    return best_mask, best_sums

def _build_min_max(deadline, seed=None):
    """Build a minimal maximal Sidon set using direct constructive approach."""
    if seed is not None:
        random.seed(seed)
    
    sel_mask = 0
    sums_mask = 0
    
    # Try random orders, build greedily, check maximality
    best_mask = None
    best_sums = None
    best_size = float('inf')
    
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        cur_sel = 0
        cur_sums = 0
        
        for x in order:
            if _bitmask_add_check(cur_sel, cur_sums, x):
                cur_sel, cur_sums = _update_masks(cur_sel, cur_sums, x)
        
        # Check if maximal
        if _is_maximal(cur_sel, cur_sums):
            sz = _count_bits(cur_sel)
            if sz < best_size:
                best_size = sz
                best_mask = cur_sel
                best_sums = cur_sums
    
    return best_mask, best_sums

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    best_mask = None
    best_sums = None
    best_size = float('inf')
    
    # Phase 1: Direct constructive search
    mask, sums = _build_min_max(deadline)
    if mask is not None:
        best_mask = mask
        best_sums = sums
        best_size = _count_bits(mask)
    
    # Phase 2: Local search to shrink
    if best_mask is not None and time.time() < deadline - 10:
        improved_mask, improved_sums = _local_search_shrink(best_mask, best_sums, deadline, max_iters=200)
        if _count_bits(improved_mask) < best_size:
            best_mask = improved_mask
            best_sums = improved_sums
            best_size = _count_bits(best_mask)
    
    # Phase 3: More trials with different seeds if time remains
    if time.time() < deadline - 20:
        for seed in range(100):
            if time.time() > deadline:
                break
            mask, sums = _build_min_max(deadline, seed=seed)
            if mask is not None:
                sz = _count_bits(mask)
                if sz < best_size:
                    best_size = sz
                    best_mask = mask
                    best_sums = sums
    
    # Final local search
    if best_mask is not None and time.time() < deadline - 5:
        improved_mask, improved_sums = _local_search_shrink(best_mask, best_sums, deadline, max_iters=300)
        if _count_bits(improved_mask) < best_size:
            best_mask = improved_mask
            best_sums = improved_sums
            best_size = _count_bits(best_mask)
    
    # Convert to list
    if best_mask is not None:
        result = [x for x in range(1, N + 1) if best_mask >> x & 1]
        return result
    
    # Fallback: simple greedy
    sel = []
    sums = set()
    for x in range(1, N + 1):
        can = True
        for a in sel:
            if x + a in sums:
                can = False
                break
        if 2 * x in sums:
            can = False
        if can:
            sel.append(x)
            for a in sel[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    return sel
