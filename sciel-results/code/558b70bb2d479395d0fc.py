"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _check_sidon(selected):
    """Check if selected is a valid Sidon set. Returns True if valid."""
    if len(selected) == 0:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
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
            b = selected[j]
            sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added
        can_add = True
        for a in selected:
            s1 = x + a
            if s1 in sums:
                can_add = False
                break
        if can_add:
            if 2 * x in sums:
                can_add = False
        if can_add:
            return False
    return True

def _blocking_score(x, selected, sums, N):
    """Score how many elements x blocks (prevents from being added later)."""
    blocked = 0
    for y in range(1, N + 1):
        if y in selected or y == x:
            continue
        # Check if y would be blocked by x being in the set
        # y is blocked if adding y to selected∪{x} creates a sum conflict
        # This happens if for some a in selected: (x+a) == (y+a') or (x+a) == 2y or 2x == (y+a) etc.
        # Simplified: check if any sum involving x conflicts with sums involving y
        # Actually, let's count how many y's would have a conflict with x
        # A conflict exists if ∃a,b in selected such that x+a = y+b, or x+a = 2y, or 2x = y+a, etc.
        # For simplicity, count direct sum conflicts
        for a in selected:
            s_xa = x + a
            # Check if this sum would conflict with any sum involving y
            for b in selected:
                s_yb = y + b
                if s_xa == s_yb:
                    blocked += 1
                    break
            else:
                if s_xa == 2 * y or 2 * x == y + a or 2 * x == 2 * y:
                    blocked += 1
                    break
    return blocked

def _greedy_with_blocking_heuristic(order, N):
    """Greedy construction using blocking heuristic for tie-breaking."""
    selected = []
    sums = set()
    
    for x in order:
        # Check if x can be added
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

def _backtrack_search(depth, max_depth, selected, sums, N, deadline):
    """Limited backtracking search for smaller maximal Sidon sets."""
    if time.time() > deadline:
        return None
    
    # Check if current set is maximal
    if _is_maximal(selected, N):
        return list(selected)
    
    if depth >= max_depth:
        return None
    
    # Find candidates that can be added
    selected_set = set(selected)
    candidates = []
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added
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
            candidates.append((x, added_sums))
    
    if not candidates:
        # No candidates can be added, but set might not be maximal
        # This shouldn't happen if we're building correctly
        return None
    
    # Try candidates in order of blocking score (higher first)
    candidates.sort(key=lambda t: -_blocking_score(t[0], selected, sums, N))
    
    for x, added_sums in candidates:
        if time.time() > deadline:
            return None
        
        new_selected = selected + [x]
        new_sums = sums | added_sums
        
        result = _backtrack_search(depth + 1, max_depth, new_selected, new_sums, N, deadline)
        if result is not None:
            return result
    
    return None

def construct():
    deadline = time.time() + 55.0  # 55s safety margin
    best_size = N + 1
    best_set = None
    
    # Try ascending greedy first (baseline)
    baseline = _greedy_with_blocking_heuristic(list(range(1, N + 1)), N)
    if _check_sidon(baseline) and _is_maximal(baseline, N):
        best_size = len(baseline)
        best_set = baseline
    
    # Random restarts with different orderings
    rng = random.Random(42)
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate a random ordering
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        candidate = _greedy_with_blocking_heuristic(order, N)
        
        if len(candidate) < best_size and _check_sidon(candidate) and _is_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break
    
    # Try limited backtracking from a few good starting points
    if best_set is not None and time.time() < deadline - 5.0:
        # Try backtracking from the best set found, trying to remove elements
        # and see if we can get a smaller maximal set
        for _ in range(10):
            if time.time() > deadline:
                break
            
            start_set = list(best_set)
            # Try removing one element at a time and see if the remainder is still maximal
            # If not, we need to add others to make it maximal again
            for i in range(len(start_set)):
                if time.time() > deadline:
                    break
                
                reduced = [start_set[j] for j in range(len(start_set)) if j != i]
                if not reduced:
                    continue
                
                # Check if reduced is a valid Sidon set
                if not _check_sidon(reduced):
                    continue
                
                # Try to extend reduced to a maximal set using greedy
                reduced_set = set(reduced)
                reduced_sums = set()
                for idx, a in enumerate(reduced):
                    for j in range(idx, len(reduced)):
                        reduced_sums.add(a + reduced[j])
                
                extended = list(reduced)
                ext_sums = reduced_sums
                for x in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if x in reduced_set:
                        continue
                    added_sums = set()
                    can_add = True
                    for a in extended:
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
                        extended.append(x)
                        ext_sums.update(added_sums)
                
                if len(extended) < best_size and _check_sidon(extended) and _is_maximal(extended, N):
                    best_size = len(extended)
                    best_set = extended
                    if best_size <= 16:
                        break
    
    if best_set is None:
        best_set = _greedy_with_blocking_heuristic(list(range(1, N + 1)), N)
    
    return best_set
# EVOLVE-BLOCK-END
