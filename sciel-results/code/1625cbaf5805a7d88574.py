"""Small inclusion-maximal Sidon sets in [1,256] with backtracking search."""
import time

N = 256
DEADLINE_FRACTION = 0.85  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all pairwise sums a+b with a<=b distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon."""
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
        new_sums_ok = True
        for a in selected:
            s = a + x
            if s in sums:
                new_sums_ok = False
                break
        if not new_sums_ok:
            continue
        # Also check 2*x
        if 2 * x in sums:
            continue
        # If we get here, adding x would preserve Sidon, so not maximal
        return False
    return True

def _greedy_saturation(selected, N):
    """Extend selected to a maximal Sidon set using greedy saturation."""
    sel = list(selected)
    sums = set()
    for i in range(len(sel)):
        for j in range(i, len(sel)):
            sums.add(sel[i] + sel[j])
    
    for x in range(1, N + 1):
        if x in sel:
            continue
        added = [x + a for a in sel] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            sel.append(x)
            sums.update(added)
    return sel

def _backtracking_search(deadline):
    """
    Backtracking search that builds a Sidon set incrementally.
    At each step, choose the candidate that blocks the most other candidates,
    to force early saturation and minimize final set size.
    
    Returns a maximal Sidon set of minimal size found within time budget.
    """
    best_result = None
    best_size = N + 1
    
    def backtrack(selected, forbidden_sums, candidate_mask, depth):
        nonlocal best_result, best_size
        
        if time.time() > deadline:
            return
        
        # If current partial set already exceeds best, prune
        if len(selected) >= best_size:
            return
        
        # Check if we can terminate: is the current set already maximal?
        # A set is maximal if every missing element is "blocked" (adding it would break Sidon)
        # An element x is blocked if:
        #   - 2x is in forbidden_sums, OR
        #   - there exists a in selected such that a+x is in forbidden_sums
        # We need to check all missing elements.
        
        sel_set = set(selected)
        is_maximal = True
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Check if x is blocked
            blocked = False
            if 2 * x in forbidden_sums:
                blocked = True
            else:
                for a in selected:
                    if (a + x) in forbidden_sums:
                        blocked = True
                        break
            if not blocked:
                is_maximal = False
                break
        
        if is_maximal:
            # Found a maximal Sidon set
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
            return
        
        # Find all valid candidates (elements that can be added without breaking Sidon)
        valid_candidates = []
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Check if adding x preserves Sidon
            ok = True
            if 2 * x in forbidden_sums:
                ok = False
            else:
                for a in selected:
                    if (a + x) in forbidden_sums:
                        ok = False
                        break
            if ok:
                valid_candidates.append(x)
        
        if not valid_candidates:
            # No valid candidates but not maximal? This shouldn't happen if logic is correct
            # But just in case, treat as maximal
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
            return
        
        # Score each candidate by how many other candidates it blocks
        # Adding x blocks y if (x+y) becomes a forbidden sum and y is currently valid
        # More precisely, adding x creates new forbidden sums: {2x} ∪ {x+a for a in selected}
        # These new sums may block other candidates y where y+2x or y+(x+a) coincides with existing sums
        # Simpler heuristic: count how many valid candidates y would become invalid after adding x
        
        # For efficiency, precompute the new sums that x would add
        def count_blocked(x):
            new_sums = {2 * x}
            new_sums.update(x + a for a in selected)
            # Count how many valid candidates y would be blocked
            blocked_count = 0
            for y in valid_candidates:
                if y == x:
                    continue
                # y is blocked if 2y in new_sums ∪ forbidden_sums, or y+a in new_sums ∪ forbidden_sums for some a in selected
                if 2 * y in new_sums or 2 * y in forbidden_sums:
                    blocked_count += 1
                    continue
                for a in selected:
                    if (y + a) in new_sums or (y + a) in forbidden_sums:
                        blocked_count += 1
                        break
            return blocked_count
        
        # Sort candidates by blocked count descending, then by value ascending
        scored = [(count_blocked(x), x) for x in valid_candidates]
        scored.sort(key=lambda t: (-t[0], t[1]))
        
        # Try each candidate
        for _, x in scored:
            if time.time() > deadline:
                return
            
            # Add x
            new_forbidden = set(forbidden_sums)
            new_forbidden.add(2 * x)
            for a in selected:
                new_forbidden.add(x + a)
            
            backtrack(selected + [x], new_forbidden, None, depth + 1)
            
            if best_size == len(selected) + 1:
                # Can't do better than this
                return
    
    # Start with empty set
    backtrack([], set(), None, 0)
    
    if best_result is not None:
        return best_result
    return None

def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1,256].
    Uses backtracking search with blocking heuristic to minimize cardinality.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_FRACTION * 60.0
    
    # Try backtracking search
    result = _backtracking_search(deadline)
    
    if result is not None:
        return result
    
    # Fallback: greedy saturation (always produces a valid maximal Sidon set)
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

# EVOLVE-BLOCK-END
