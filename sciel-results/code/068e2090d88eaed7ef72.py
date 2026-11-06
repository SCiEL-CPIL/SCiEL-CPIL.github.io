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
        if len(unblocked) > 40:
            candidates = rng.sample(unblocked, 40)
        else:
            candidates = list(unblocked)
            rng.shuffle(candidates)
            
        best_x = None
        best_score = float('inf')
        best_tie = None
        
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
            
            # Score: number of unblocked elements this x would block
            score = 0
            for y in unblocked:
                if y == x:
                    continue
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
                    if y + x in new_sums:
                        y_blocked = True
                if y_blocked:
                    score += 1
            
            # We want to MAXIMIZE the number blocked, so minimize -score
            # Actually, we want to pick the x that blocks the MOST unblocked elements
            # So we want to MAXIMIZE score. Let's invert: score is what we want to max.
            if score > best_score if best_score != float('inf') else False:
                if best_score == float('inf'):
                    best_score = score
                    best_x = x
                elif score > best_score:
                    best_score = score
                    best_x = x
                    best_tie = None
            elif best_x is not None and score == best_score:
                if best_tie is None:
                    best_tie = x
                # Tie break: prefer larger x? or random
                if rng.random() < 0.5:
                    best_tie = x
            if best_x is None:
                best_x = x
                best_score = score

        # Fix logic: we want max score. 
        # Let's re-implement the selection logic clearly.
        best_x = None
        max_blocks = -1
        ties = []
        
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
                if not y_blocked:
                    if y + x in new_sums:
                        y_blocked = True
                if y_blocked:
                    score += 1
            
            if score > max_blocks:
                max_blocks = score
                best_x = x
                ties = []
            elif score == max_blocks:
                ties.append(x)
        
        if best_x is None:
            if candidates:
                best_x = rng.choice(candidates)
            else:
                break
        elif ties:
            best_x = rng.choice(ties)
        
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
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
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

def _construct():
    start_time = time.time()
    # Leave 2 seconds buffer for Python to exit
    deadline = start_time + 58.0
    
    best = None
    best_size = float('inf')
    
    seeds = [42, 123, 999, 2024, 314, 1618, 777, 555, 888, 444, 111, 222, 333, 500, 600, 700]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        
        # Try a few initial configurations per seed
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            # Start with a small random Sidon set
            # Smaller start might lead to more efficient blocking?
            target_size = rng.randint(4, 15)
            for x in candidates:
                if len(selected) >= target_size:
                    break
                ok, selected, sums = _try_add(selected, sums, x)
            
            if not selected:
                continue
            
            # Maximize
            selected, sums = _greedy_maximize(selected, sums, rng, deadline)
            
            # Validate
            if not _is_valid_sidon(selected):
                continue
            if not _is_maximal(selected, sums):
                continue
            
            # Update best
            if len(selected) < best_size:
                best_size = len(selected)
                best = list(selected)
            
            # Refine
            if time.time() < deadline and best_size > 10:
                selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                    if len(selected2) < best_size:
                        best_size = len(selected2)
                        best = list(selected2)
            
            # If we found a very small set, we can stop early
            if best_size <= 10:
                break
    
    if best is None:
        # Fallback: deterministic greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best

def construct():
    return _construct()
