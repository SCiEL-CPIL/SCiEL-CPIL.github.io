"""Backtracking/greedy blocking-priority search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check all unordered pair sums a+b (a<=b) are distinct."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return True

def _blocked_mask(selected):
    """
    Return a bitmask (Python int) of elements in [1,N] that CANNOT be added
    to 'selected' while preserving the Sidon property.
    An element x is blocked if adding it creates a duplicate pair-sum.
    """
    # Precompute existing sums
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    
    blocked = 0
    for x in range(1, N + 1):
        if x in selected:
            continue
        # Check if adding x creates a duplicate sum
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s = 2 * x
            if s in sums:
                ok = False
        if not ok:
            blocked |= (1 << x)
    return blocked

def _greedy_blocking(selected, used_sums, rng, time_limit):
    """
    Greedy: repeatedly pick the addable element that blocks the most new candidates.
    Returns (final_set, final_sums) or None if time exceeded.
    """
    selected = list(selected)
    used_sums = set(used_sums)
    
    # Precompute initial blocked
    # We'll track which elements are still addable
    addable = set(range(1, N + 1))
    for x in selected:
        addable.discard(x)
    
    # Determine which addable elements are actually blocked
    def is_blocked(x):
        for a in selected:
            if x + a in used_sums:
                return True
        if 2 * x in used_sums:
            return True
        return False
    
    # Remove blocked from addable
    addable = {x for x in addable if not is_blocked(x)}
    
    while addable:
        if time.time() > time_limit:
            return None
        
        # Score each candidate: how many currently-addable elements would become blocked
        best_x = None
        best_score = -1
        best_newly_blocked = set()
        
        candidates = list(addable)
        rng.shuffle(candidates)  # random tie-breaking
        
        for x in candidates:
            # Simulate adding x
            new_sums = set()
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            # Check no conflict
            if new_sums & used_sums:
                continue  # shouldn't happen since x is addable
            
            # Count how many currently-addable elements become blocked
            newly_blocked = set()
            for y in addable:
                if y == x:
                    continue
                # Check if y would be blocked after adding x
                blocked = False
                for a in selected:
                    if y + a in used_sums:
                        blocked = True
                        break
                if not blocked:
                    if 2 * y in used_sums:
                        blocked = True
                # Also check against new sums from x
                if not blocked:
                    for a in selected:
                        if y + a in new_sums:
                            blocked = True
                            break
                if not blocked:
                    if y + x in used_sums:
                        blocked = True
                    elif 2 * y in new_sums:
                        blocked = True
                if blocked:
                    newly_blocked.add(y)
            
            score = len(newly_blocked)
            if score > best_score:
                best_score = score
                best_x = x
                best_newly_blocked = newly_blocked
        
        if best_x is None:
            # All candidates are somehow blocked; pick any
            best_x = candidates[0] if candidates else None
            if best_x is None:
                break
            # Just add it
            used_sums.update([best_x + a for a in selected] + [2 * best_x])
            selected.append(best_x)
            addable.discard(best_x)
            # Recompute addable
            new_addable = set()
            for x in addable:
                blocked = False
                for a in selected:
                    if x + a in used_sums:
                        blocked = True
                        break
                if not blocked and 2 * x in used_sums:
                    blocked = True
                if not blocked:
                    new_addable.add(x)
            addable = new_addable
            continue
        
        # Add best_x
        used_sums.update([best_x + a for a in selected] + [2 * best_x])
        selected.append(best_x)
        addable.discard(best_x)
        
        # Remove newly blocked elements from addable
        addable -= best_newly_blocked
        
        # Also remove any that are now blocked (recheck)
        still_addable = set()
        for x in addable:
            blocked = False
            for a in selected:
                if x + a in used_sums:
                    blocked = True
                    break
            if not blocked and 2 * x in used_sums:
                blocked = True
            if not blocked:
                still_addable.add(x)
        addable = still_addable
    
    return selected, used_sums

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    if not _is_sidon(selected):
        return False
    sel_set = set(selected)
    # Compute all sums
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    # Try every missing element
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x preserves Sidon
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok and 2 * x in sums:
            ok = False
        if ok:
            return False  # x could be added, so not maximal
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit
    
    # Get initial upper bound from simple greedy
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)
    
    best_set = simple_selected
    best_size = len(best_set)
    
    # Verify it's maximal (it should be by construction, but check)
    if not _verify_maximal(best_set):
        # Force maximality by adding all possible
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in best_set:
                    added = [x + a for a in best_set] + [2 * x]
                    if len(set(added)) == len(added) and not any(s in simple_sums for s in added):
                        best_set.append(x)
                        simple_sums.update(added)
                        changed = True
    
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Try multiple randomized greedy blocking strategies
    for attempt in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Start with a random subset of small elements to seed
        seed_size = rng.randint(0, 3)
        seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
        
        # Check seed is Sidon
        seed_ok = True
        seed_sums = set()
        for i, a in enumerate(seed_elements):
            for b in seed_elements[i:]:
                s = a + b
                if s in seed_sums:
                    seed_ok = False
                    break
                seed_sums.add(s)
            if not seed_ok:
                break
        
        if not seed_ok:
            continue
        
        result = _greedy_blocking(seed_elements, seed_sums, rng, deadline - 1.0)
        if result is None:
            continue
        
        final_set, final_sums = result
        
        # Verify
        if not _verify_maximal(final_set):
            continue
        
        if len(final_set) < best_size:
            best_size = len(final_set)
            best_set = final_set
    
    return best_set
# EVOLVE-BLOCK-END
