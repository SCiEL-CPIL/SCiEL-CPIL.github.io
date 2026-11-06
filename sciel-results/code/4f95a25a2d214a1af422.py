"""Small inclusion-maximal Sidon sets in [1,256]"""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b with a<=b distinct)."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            total = a + b
            if total in s:
                return False
            s.add(total)
    return True

def _compute_sums(selected):
    """Compute the set of all a+b sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to the Sidon set."""
    for a in selected:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def _greedy_build(order, deadline):
    """Build a Sidon set by processing elements in the given order.
    Returns (selected, sums) where sums is the set of all a+b sums."""
    selected = []
    sums = set()
    for x in order:
        if time.time() > deadline:
            break
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            selected.append(x)
            for a in selected:
                if a != x:
                    sums.add(x + a)
            sums.add(2 * x)
    return selected, sums

def _maximalize(selected, sums, deadline):
    """Make the set maximal by adding elements that don't break Sidon.
    Returns (new_selected, new_sums)."""
    new_selected = list(selected)
    new_sums = set(sums)
    sel_set = set(new_selected)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if _is_addable(x, new_selected, new_sums):
                new_selected.append(x)
                sel_set.add(x)
                for a in new_selected:
                    if a != x:
                        new_sums.add(x + a)
                new_sums.add(2 * x)
                changed = True
                break
    
    return new_selected, new_sums

def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            return False
    return True

def _local_search_reduce(selected, sums, deadline):
    """Try to reduce the size of a maximal Sidon set via removal.
    Iteratively removes an element, re-maximalizes, and keeps the result
    if it is smaller. Repeats until no improvement is possible or time runs out.
    Returns (new_selected, new_sums)."""
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for y in list(best_sel):
            if time.time() > deadline:
                break
            if len(best_sel) <= 1:
                break
            
            # Remove y
            candidate = [x for x in best_sel if x != y]
            cand_sums = _compute_sums(candidate)
            
            # Re-maximalize
            cand_sel, cand_sums = _maximalize(candidate, cand_sums, deadline)
            
            # Check if it's actually smaller
            if len(cand_sel) < best_size:
                if _is_sidon(cand_sel) and _is_maximal(cand_sel, cand_sums):
                    best_sel = cand_sel
                    best_sums = cand_sums
                    best_size = len(best_sel)
                    improved = True
                    break
    
    return best_sel, best_sums

def _swap_search(selected, sums, deadline):
    """Try swapping elements: remove one, add another, re-maximalize.
    Returns (new_selected, new_sums)."""
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    sel_set = set(best_sel)
    
    for y in list(best_sel):
        if time.time() > deadline:
            break
        if len(best_sel) <= 1:
            break
        
        # Remove y
        candidate = [x for x in best_sel if x != y]
        cand_sums = _compute_sums(candidate)
        
        # Try adding each missing element
        cand_set = set(candidate)
        for z in range(1, N + 1):
            if z in cand_set or z == y:
                continue
            if _is_addable(z, candidate, cand_sums):
                new_candidate = candidate + [z]
                new_sums = _compute_sums(new_candidate)
                # Re-maximalize
                new_sel, new_sums = _maximalize(new_candidate, new_sums, deadline)
                if len(new_sel) < best_size:
                    if _is_sidon(new_sel) and _is_maximal(new_sel, new_sums):
                        best_sel = new_sel
                        best_sums = new_sums
                        best_size = len(best_sel)
                        sel_set = set(best_sel)
    
    return best_sel, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin before 60s kill
    
    # Strategy: try many random orderings, keep the best (smallest) maximal set
    best_selected = None
    best_size = N + 1
    
    # First, try the standard ascending greedy as baseline
    selected, sums = _greedy_build(range(1, N + 1), deadline)
    if selected:
        selected, sums = _maximalize(selected, sums, deadline)
        if _is_sidon(selected) and _is_maximal(selected, sums):
            selected, sums = _local_search_reduce(selected, sums, deadline)
            if _is_sidon(selected) and _is_maximal(selected, sums):
                best_selected = selected
                best_size = len(selected)
    
    # Now try random orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_build(order, deadline)
        if not selected:
            continue
        
        selected, sums = _maximalize(selected, sums, deadline)
        
        if len(selected) >= best_size:
            continue
        
        if _is_sidon(selected) and _is_maximal(selected, sums):
            selected, sums = _local_search_reduce(selected, sums, deadline)
            if _is_sidon(selected) and _is_maximal(selected, sums) and len(selected) < best_size:
                best_selected = selected
                best_size = len(selected)
    
    # Final polish: try swap search on the best solution
    if best_selected is not None and time.time() < deadline:
        best_sums = _compute_sums(best_selected)
        best_selected, best_sums = _swap_search(best_selected, best_sums, deadline)
        if _is_sidon(best_selected) and _is_maximal(best_selected, best_sums):
            # Try local search again after swaps
            best_selected, best_sums = _local_search_reduce(best_selected, best_sums, deadline)
            if _is_sidon(best_selected) and _is_maximal(best_selected, best_sums):
                pass  # Keep it
    
    if best_selected is None:
        # Fallback: return the ascending greedy result
        selected, sums = _greedy_build(range(1, N + 1), deadline)
        selected, sums = _maximalize(selected, sums, deadline)
        best_selected = selected
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
