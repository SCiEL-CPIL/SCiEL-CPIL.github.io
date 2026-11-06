"""Small inclusion-maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all sums a+b for a<=b are distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal: no missing element can be added."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x preserves Sidon property
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if s2 in sums:
                ok = False
        if ok:
            return False
    return True

def _validate(selected):
    """Full validation: Sidon + maximal."""
    if not selected:
        return False
    if len(set(selected)) != len(selected):
        return False
    for x in selected:
        if x < 1 or x > N:
            return False
    return _is_sidon(selected) and _is_maximal(selected)

def _greedy_saturation(starting_set=None, order=None):
    """
    Greedy saturation: given an optional starting Sidon set, greedily add
    elements (in the given order or default ascending) that preserve Sidon,
    until no more can be added. This produces a maximal Sidon set.
    """
    if starting_set is None:
        selected = []
    else:
        selected = list(starting_set)
    
    if not _is_sidon(selected):
        return None
    
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    sel_set = set(selected)
    
    candidates = list(range(1, N + 1))
    if order is not None:
        # Use custom order
        candidates = [x for x in order if x not in sel_set]
    else:
        candidates = [x for x in candidates if x not in sel_set]
    
    for x in candidates:
        if x in sel_set:
            continue
        # Check if x can be added
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if s2 in sums:
                ok = False
        if ok:
            selected.append(x)
            sel_set.add(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    return selected

def _try_remove_to_shrink(selected, deadline):
    """
    Try to remove elements from a maximal Sidon set to get a smaller maximal one.
    Strategy: try removing each element; if the result is still Sidon, check if
    it can be made maximal again by adding other elements. If the resulting
    maximal set is smaller, keep it.
    """
    current = list(selected)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current)):
            if time.time() > deadline:
                break
            candidate = current[:i] + current[i+1:]
            if not _is_sidon(candidate):
                continue
            # Try to saturate to get a maximal set
            saturated = _greedy_saturation(starting_set=candidate)
            if saturated is None:
                continue
            if len(saturated) < len(current):
                current = saturated
                improved = True
                break
    return current

def _random_order():
    """Generate a random permutation of [1, N]."""
    order = list(range(1, N + 1))
    random.shuffle(order)
    return order

def _greedy_with_random_order(order, deadline):
    """Greedy saturation with a specific order."""
    selected = []
    sums = set()
    sel_set = set()
    
    for x in order:
        if x in sel_set:
            continue
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if s2 in sums:
                ok = False
        if ok:
            selected.append(x)
            sel_set.add(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    return selected

def _local_search_shrink(selected, deadline):
    """
    More aggressive local search: try all single removals, then for each
    that yields a valid Sidon set, try to find a maximal extension that is
    smaller. Also try swapping one element for another.
    """
    current = list(selected)
    best = current
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current)):
            if time.time() > deadline:
                break
            # Try removing element i
            candidate = current[:i] + current[i+1:]
            if not _is_sidon(candidate):
                continue
            # Saturate to get maximal
            saturated = _greedy_saturation(starting_set=candidate)
            if saturated is not None and len(saturated) < len(best):
                best = saturated
                current = saturated
                improved = True
                break
    return best

def construct():
    """
    Find a small inclusion-maximal Sidon set in [1, 256].
    Strategy:
    1. Run greedy saturation with multiple random orders to get various maximal sets.
    2. For each, apply local search to shrink by removing elements and re-saturating.
    3. Keep the smallest valid maximal Sidon set found.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    random.seed(42)
    
    best_set = None
    best_size = N + 1
    
    # First: default ascending greedy
    asc_result = _greedy_saturation()
    if asc_result and _validate(asc_result):
        best_set = asc_result
        best_size = len(asc_result)
    
    # Try multiple random orders
    num_trials = 300
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = _random_order()
        result = _greedy_with_random_order(order, deadline)
        
        if result is None or not _is_sidon(result):
            continue
        
        # Check maximality
        if not _is_maximal(result):
            # Saturate it
            result = _greedy_saturation(starting_set=result)
            if result is None:
                continue
        
        if len(result) < best_size:
            if _validate(result):
                best_set = result
                best_size = len(result)
        
        # Local search to shrink
        if time.time() < deadline:
            shrunk = _local_search_shrink(result, deadline)
            if shrunk and len(shrunk) < best_size:
                if _validate(shrunk):
                    best_set = shrunk
                    best_size = len(shrunk)
    
    # Additional targeted attempts: try to build small sets from scratch
    # by starting with a small seed and saturating
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick a random small subset as seed
            seed_size = random.randint(3, 8)
            seed = random.sample(range(1, N + 1), seed_size)
            # Check if seed is Sidon
            if not _is_sidon(seed):
                continue
            result = _greedy_saturation(starting_set=seed)
            if result is None:
                continue
            if not _is_maximal(result):
                result = _greedy_saturation(starting_set=result)
                if result is None:
                    continue
            if len(result) < best_size:
                if _validate(result):
                    best_set = result
                    best_size = len(result)
            if time.time() < deadline:
                shrunk = _local_search_shrink(result, deadline)
                if shrunk and len(shrunk) < best_size:
                    if _validate(shrunk):
                        best_set = shrunk
                        best_size = len(shrunk)
    
    if best_set is None:
        # Fallback: ascending greedy
        best_set = _greedy_saturation()
    
    return best_set

# EVOLVE-BLOCK-END
