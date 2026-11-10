"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if not selected:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]. Assumes valid Sidon."""
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        can_add = True
        for a in selected:
            if x + a in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        if can_add:
            return False
    return True

def _greedy_build(order, N):
    """Greedy construction: add elements in given order if they don't conflict."""
    selected = []
    sums = set()
    for x in order:
        added_sums = set()
        can_add = True
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            selected.append(x)
            sums.update(added_sums)
    return selected

def _greedy_extend(base, N):
    """Greedily extend a valid Sidon set to maximal in [1,N]."""
    ext = list(base)
    ext_set = set(ext)
    ext_sums = set()
    for i, a in enumerate(ext):
        for j in range(i, len(ext)):
            ext_sums.add(a + ext[j])
    
    for x in range(1, N + 1):
        if x in ext_set:
            continue
        added_sums = set()
        can_add = True
        for a in ext:
            s = x + a
            if s in ext_sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in ext_sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            ext.append(x)
            ext_set.add(x)
            ext_sums.update(added_sums)
    return ext

def _count_removable(selected, N):
    """Count how many elements can be removed while keeping a valid Sidon set."""
    n = len(selected)
    count = 0
    for i in range(n):
        reduced = [selected[j] for j in range(n) if j != i]
        if _is_valid_sidon(reduced):
            count += 1
    return count

def _try_remove_and_extend(selected, N):
    """
    Try removing one element and greedily extending to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 1:
        return None
    
    best = None
    best_len = n + 1
    
    # Sort removal candidates by heuristic: prefer removing elements that
    # are "central" (have many conflicts) to maximize room for re-extension
    indices = list(range(n))
    
    for i in indices:
        if time.time() > 58.0:
            break
        reduced = [selected[j] for j in range(n) if j != i]
        if not reduced:
            continue
        if not _is_valid_sidon(reduced):
            continue
        
        # Greedily extend reduced to maximal
        ext = _greedy_extend(reduced, N)
        
        if _is_valid_sidon(ext) and _is_maximal(ext, N):
            if len(ext) < best_len:
                best_len = len(ext)
                best = list(ext)
    
    return best

def _try_swap(selected, N):
    """
    Try swapping one element out and another in, then extend to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 2:
        return None
    
    best = None
    best_len = n + 1
    selected_set = set(selected)
    
    for i in range(n):
        if time.time() > 58.0:
            break
        removed = selected[i]
        reduced = [selected[j] for j in range(n) if j != i]
        if not _is_valid_sidon(reduced):
            continue
        
        reduced_set = set(reduced)
        reduced_sums = set()
        for i2, a in enumerate(reduced):
            for j2 in range(i2, len(reduced)):
                reduced_sums.add(a + reduced[j2])
        
        # Find candidates that can replace 'removed'
        for x in range(1, N + 1):
            if time.time() > 58.0:
                break
            if x in reduced_set or x == removed:
                continue
            can_add = True
            for a in reduced:
                if x + a in reduced_sums:
                    can_add = False
                    break
            if can_add and 2 * x in reduced_sums:
                can_add = False
            if not can_add:
                continue
            
            # Build new set with x instead of removed
            new_base = reduced + [x]
            new_set = _greedy_extend(new_base, N)
            
            if _is_valid_sidon(new_set) and _is_maximal(new_set, N):
                if len(new_set) < best_len:
                    best_len = len(new_set)
                    best = list(new_set)
    
    return best

def _try_double_remove(selected, N):
    """
    Try removing two elements and greedily extending to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 3:
        return None
    
    best = None
    best_len = n + 1
    
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > 58.0:
                break
            reduced = [selected[k] for k in range(n) if k != i and k != j]
            if len(reduced) < 2:
                continue
            if not _is_valid_sidon(reduced):
                continue
            
            ext = _greedy_extend(reduced, N)
            
            if _is_valid_sidon(ext) and _is_maximal(ext, N):
                if len(ext) < best_len:
                    best_len = len(ext)
                    best = list(ext)
    
    return best

def construct():
    deadline = time.time() + 55.0  # safety margin
    rng = random.Random(42)
    
    best_set = None
    best_size = N + 1
    
    # Phase 1: Diverse greedy starts
    baseline = _greedy_build(list(range(1, N + 1)), N)
    if _is_valid_sidon(baseline) and _is_maximal(baseline, N):
        best_size = len(baseline)
        best_set = list(baseline)
    
    desc = _greedy_build(list(range(N, 0, -1)), N)
    if _is_valid_sidon(desc) and _is_maximal(desc, N):
        if len(desc) < best_size:
            best_size = len(desc)
            best_set = list(desc)
    
    # Random restarts with diverse orderings
    for attempt in range(500):
        if time.time() > deadline - 20.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if _is_valid_sidon(candidate) and _is_maximal(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)
                if best_size <= 14:
                    break
    
    if best_set is None:
        best_set = baseline
        best_size = len(best_set)
    
    # Phase 2: Local improvement via removal, swap, and double-removal
    current = list(best_set)
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Try single removal
        result = _try_remove_and_extend(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            if len(current) < best_size:
                best_size = len(current)
                best_set = list(current)
        
        # Try swap
        result = _try_swap(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            if len(current) < best_size:
                best_size = len(current)
                best_set = list(current)
        
        # Try double removal
        result = _try_double_remove(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            if len(current) < best_size:
                best_size = len(current)
                best_set = list(current)
    
    # Phase 3: More random restarts with local improvement
    for attempt in range(300):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if not (_is_valid_sidon(candidate) and _is_maximal(candidate, N)):
            continue
        if len(candidate) >= best_size:
            continue
        
        cur = list(candidate)
        imp = True
        while imp and time.time() < deadline - 3.0:
            imp = False
            r = _try_remove_and_extend(cur, N)
            if r is not None and len(r) < len(cur):
                cur = r
                imp = True
            r = _try_swap(cur, N)
            if r is not None and len(r) < len(cur):
                cur = r
                imp = True
            r = _try_double_remove(cur, N)
            if r is not None and len(r) < len(cur):
                cur = r
                imp = True
        
        if len(cur) < best_size:
            best_size = len(cur)
            best_set = list(cur)
            if best_size <= 14:
                break
    
    return best_set
# EVOLVE-BLOCK-END
