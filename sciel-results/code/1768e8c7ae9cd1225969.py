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
            sums.add(a + selected[j])
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
        
        # If unblocked list is large, sample to speed up scoring
        if len(unblocked) > 30:
            candidates = rng.sample(unblocked, 30)
        else:
            candidates = unblocked
            
        best_x = None
        best_score = -1
        
        for x in candidates:
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
            # Count how many unblocked elements this x would block
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is blocked by x in the new set
                # y is blocked if 2y in new_sums OR (y+a) in new_sums for some a in selected+{x}
                if 2 * y in new_sums:
                    score += 1
                    continue
                
                y_blocked = False
                for a in selected:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if not y_blocked:
                    # Check if y+x is in new_sums (which includes x+x and x+a)
                    # Actually, we just need to check if adding x blocks y
                    # The condition is: exists a' in new_selected such that y+a' in new_sums
                    # We already checked a in selected.
                    # Now check a'=x.
                    # Is y+x in new_sums?
                    # new_sums contains sums from selected+{x}.
                    # y+x is a sum of y (not in set) and x (in set).
                    # Wait, the blocking condition is y+a' in Sigma(new_set).
                    # Sigma(new_set) contains sums of elements in new_set.
                    # y is NOT in new_set.
                    # So y+x is NOT in Sigma(new_set) unless y is in new_set? No.
                    # The condition is: y is blocked if y+a' in Sigma(new_set) for some a' in new_set.
                    # If a'=x, we check if y+x in Sigma(new_set).
                    # Sigma(new_set) contains sums u+v where u,v in new_set.
                    # y+x is not necessarily a sum of two elements in new_set.
                    # It is a sum of y (external) and x (internal).
                    # The definition of blocked:
                    # x is blocked if 2x in Sigma(S) or exists a in S such that x+a in Sigma(S).
                    # So for y to be blocked by x (as part of S), we need y+x in Sigma(S).
                    # Sigma(S) is the set of pair sums of S.
                    # So we check if y+x is equal to some u+v where u,v in S.
                    
                    if y + x in new_sums:
                        y_blocked = True
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            # Fallback: pick random unblocked
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
        # Shuffle to avoid bias
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            # Recompute sums for candidate
            new_sums = set()
            valid = True
            n = len(candidate)
            for a_idx in range(n):
                a = candidate[a_idx]
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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = None
    best_size = float('inf')
    
    # Use multiple seeds for diversity
    seeds = [42, 123, 999, 2024, 314, 1618]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        
        # Strategy: Random initial Sidon set, then greedy maximize
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            # Start with a small random Sidon set
            target_size = rng.randint(5, 15)
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
            
            if best_size <= 12:
                break
    
    if best is None:
        # Fallback: deterministic greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
