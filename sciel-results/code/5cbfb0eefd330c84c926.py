import random
import time

N = 256

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal.
    For every x not in selected, adding x must create a conflict.
    x conflicts iff 2x in sums OR exists a in selected such that x+a in sums.
    """
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x is blocked
        if 2 * x in sums:
            continue
        # Check if x + a in sums for any a in selected
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _conflict_count(selected, sums):
    """Count how many missing elements are NOT blocked (i.e., can be added)."""
    sel_set = set(selected)
    count = 0
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
            count += 1
    return count

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

def _try_add(selected, sums, x):
    """Try to add x to selected. Return (success, new_selected, new_sums)."""
    new_sums = set(sums)
    # Check 2x
    if 2 * x in new_sums:
        return False, selected, sums
    new_sums.add(2 * x)
    # Check x + a for all a in selected
    for a in selected:
        s = x + a
        if s in new_sums:
            return False, selected, sums
        new_sums.add(s)
    new_selected = selected + [x]
    return True, new_selected, new_sums

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, trying to minimize additions."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        # Score each candidate by how many other unblocked elements it would block
        best_x = None
        best_score = -1
        for x in unblocked:
            # Simulate adding x
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
            
            # Count how many other unblocked elements are now blocked
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is now blocked with new_sums
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected + [x]:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            # Fallback: pick random
            best_x = rng.choice(unblocked)
        
        # Add best_x
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            # Shouldn't happen, but break to avoid infinite loop
            break
    
    return selected, sums

def _local_search(selected, sums, rng, deadline):
    """Try to remove elements while maintaining maximality, or swap to reduce size."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            # Recompute sums
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
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Multiple strategies
    # Strategy 1: Random initial Sidon set, then greedy maximize
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random small Sidon set
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        # Try to build a small Sidon set of size ~15-25
        target_size = rng.randint(12, 25)
        for x in candidates:
            if len(selected) >= target_size:
                break
            ok, selected, sums = _try_add(selected, sums, x)
            if ok:
                pass
            else:
                continue
        
        if not selected:
            continue
        
        # Now greedily maximize
        selected, sums = _greedy_maximize(selected, sums, rng, deadline)
        
        # Verify
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        # Local search to try to reduce size
        if time.time() < deadline:
            selected2, sums2 = _local_search(selected, sums, rng, deadline)
            if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                if len(selected2) < best_size:
                    best_size = len(selected2)
                    best = list(selected2)
        
        # Early termination if we find a very small set
        if best_size <= 12:
            break
    
    # Strategy 2: Start from known small structures
    if time.time() < deadline:
        # Try a structured approach: use elements spaced out
        for base in range(1, 10):
            if time.time() > deadline:
                break
            selected = []
            sums = set()
            # Add elements with large gaps
            for x in range(base, N + 1, base * 2 + 1):
                ok, selected, sums = _try_add(selected, sums, x)
                if ok:
                    continue
            if selected:
                selected, sums = _greedy_maximize(selected, sums, rng, deadline)
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                    if time.time() < deadline:
                        selected2, sums2 = _local_search(selected, sums, rng, deadline)
                        if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                            if len(selected2) < best_size:
                                best_size = len(selected2)
                                best = list(selected2)
    
    if best is None:
        # Fallback: use the original greedy approach
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
