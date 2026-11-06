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
    """Check if selected is inclusion-maximal using fast conflict-set computation."""
    sel_set = set(selected)
    # Compute conflict set: x is blocked if 2x in sums OR exists a in selected with x+a in sums
    # x is unblocked iff x not in sel_set AND x not in conflict_set
    conflict = set()
    # From 2x in sums: x = s/2 for even s in sums
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                conflict.add(x)
    # From x+a in sums: for each a in selected, x = s - a for s in sums
    for a in selected:
        for s in sums:
            x = s - a
            if 1 <= x <= N:
                conflict.add(x)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if x not in conflict:
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

def _get_unblocked_fast(selected, sums):
    """Return list of missing elements that can be added, using fast conflict-set computation."""
    sel_set = set(selected)
    conflict = set()
    # From 2x in sums
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                conflict.add(x)
    # From x+a in sums
    for a in selected:
        for s in sums:
            x = s - a
            if 1 <= x <= N:
                conflict.add(x)
    unblocked = []
    for x in range(1, N + 1):
        if x not in sel_set and x not in conflict:
            unblocked.append(x)
    return unblocked

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, trying to minimize additions."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked_fast(selected, sums)
        if not unblocked:
            break
        
        # If unblocked list is large, sample to speed up scoring
        if len(unblocked) > 20:
            candidates = rng.sample(unblocked, 20)
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
            
            # Score: count how many currently-unblocked elements would become blocked
            # Fast check: y is blocked by new set if 2y in new_sums or exists a' in selected+{x} with y+a' in new_sums
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                # Check y+a in new_sums for a in selected
                blocked = False
                for a in selected:
                    if y + a in new_sums:
                        blocked = True
                        break
                if not blocked:
                    if y + x in new_sums:
                        blocked = True
                if blocked:
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

def _swap_search(selected, sums, rng, deadline):
    """Try removing elements to reduce size while maintaining maximality."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            new_sums = set()
            valid = True
            m = len(candidate)
            for a_idx in range(m):
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

def _hit_set_search(selected, sums, rng, deadline):
    """Use hit set approach to find minimal maximal Sidon sets."""
    selected = list(selected)
    sums = set(sums)
    
    # Try to find a smaller maximal set by starting from a random subset
    for attempt in range(15):
        if time.time() > deadline:
            break
        
        # Randomly pick a subset of current solution
        k = max(2, len(selected) // 2)
        subset = rng.sample(selected, k)
        subset.sort()
        
        # Try to extend this subset to a maximal set
        sub_sums = _compute_sums(subset)
        extended, ext_sums = _greedy_maximize(subset, sub_sums, rng, deadline)
        
        if not _is_valid_sidon(extended):
            continue
        if not _is_maximal(extended, ext_sums):
            continue
        
        if len(extended) < len(selected):
            selected = extended
            sums = ext_sums
    
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = None
    best_size = float('inf')
    
    # Use multiple seeds for diversity
    seeds = [42, 123, 999, 2024, 314, 1618, 777, 555, 888, 444, 1337, 2023, 7, 11, 13, 17, 19, 23, 29, 31]
    
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
            target_size = rng.randint(3, 12)
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
                    selected3, sums3 = _swap_search(selected2, sums2, rng, deadline)
                    if _is_valid_sidon(selected3) and _is_maximal(selected3, sums3):
                        if len(selected3) < best_size:
                            best_size = len(selected3)
                            best = list(selected3)
                
                if time.time() < deadline:
                    selected4, sums4 = _hit_set_search(selected3, sums3, rng, deadline)
                    if _is_valid_sidon(selected4) and _is_maximal(selected4, sums4):
                        if len(selected4) < best_size:
                            best_size = len(selected4)
                            best = list(selected4)
            
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
