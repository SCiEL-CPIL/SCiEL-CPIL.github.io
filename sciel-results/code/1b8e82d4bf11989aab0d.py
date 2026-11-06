import random
import time

N = 256

def _compute_sums(selected):
    """Compute all unordered pair sums (including a+a) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            sums.add(a + b)
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _try_add(selected, sums, x):
    """Try to add x to selected. Return (success, new_selected, new_sums)."""
    if 2 * x in sums:
        return False, selected, sums
    for a in selected:
        if x + a in sums:
            return False, selected, sums
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
    new_selected = selected + [x]
    return True, new_selected, new_sums

def _get_unblocked(selected, sums):
    """Return list of missing elements that can be added (not blocked)."""
    sel_set = set(selected)
    unblocked = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, trying to minimize additions."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if not valid:
                continue
            
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if not y_blocked and y + x in new_sums:
                    y_blocked = True
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            best_x = rng.choice(unblocked)
        
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            break
    
    return selected, sums

def _local_search_reduce(selected, sums, rng, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            new_sums = set()
            valid = True
            for a_idx, a in enumerate(candidate):
                for b in candidate[a_idx:]:
                    s = a + b
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def _hit_set_search(selected, sums, rng, deadline):
    """
    Model as minimum hitting set: for each missing element x, we need to block it.
    x is blocked if 2x is in sums OR x+a is in sums for some a in selected.
    We try to find a smaller set by identifying which elements are "essential" for blocking.
    """
    selected = list(selected)
    sums = set(sums)
    sel_set = set(selected)
    
    # For each missing x, find which selected elements contribute to blocking it
    # If removing an element doesn't unblock any x, it's safe to remove
    # If removing an element unblocks some x, we need to check if other elements still block those x
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            new_sums = set()
            valid = True
            for a_idx, a in enumerate(candidate):
                for b in candidate[a_idx:]:
                    s = a + b
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Random initial Sidon set, then greedy maximize
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        target_size = rng.randint(8, 20)
        for x in candidates:
            if len(selected) >= target_size:
                break
            ok, selected, sums = _try_add(selected, sums, x)
        
        if not selected:
            continue
        
        selected, sums = _greedy_maximize(selected, sums, rng, deadline)
        
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        if time.time() < deadline:
            selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
            if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                if len(selected2) < best_size:
                    best_size = len(selected2)
                    best = list(selected2)
            
            if time.time() < deadline:
                selected3, sums3 = _hit_set_search(selected2, sums2, rng, deadline)
                if _is_valid_sidon(selected3) and _is_maximal(selected3, sums3):
                    if len(selected3) < best_size:
                        best_size = len(selected3)
                        best = list(selected3)
        
        if best_size <= 10:
            break
    
    # Strategy 2: Structured approach
    if time.time() < deadline:
        for base in range(1, 15):
            if time.time() > deadline:
                break
            selected = []
            sums = set()
            for x in range(base, N + 1, base * 2 + 1):
                ok, selected, sums = _try_add(selected, sums, x)
            if selected:
                selected, sums = _greedy_maximize(selected, sums, rng, deadline)
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                    if time.time() < deadline:
                        selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                        if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                            if len(selected2) < best_size:
                                best_size = len(selected2)
                                best = list(selected2)
    
    if best is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
