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

def _compute_sums(selected):
    """Compute all pair sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to selected while preserving Sidon property."""
    if x in selected:
        return False
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_fast(selected, sums):
    """Check if selected is inclusion-maximal (no element can be added)."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            return False
    return True

def _greedy_construct(seed_elements, rng, time_limit):
    """
    Greedy construction: repeatedly pick the addable element that blocks the most candidates.
    Returns (final_set, final_sums) or None if time exceeded.
    """
    selected = list(seed_elements)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    
    # Determine which elements are currently addable
    addable = set()
    for x in range(1, N + 1):
        if x not in sel_set and _is_addable(x, selected, sums):
            addable.add(x)
    
    while addable:
        if time.time() > time_limit:
            return None
        
        # Score each candidate: how many currently-addable elements would become blocked
        best_x = None
        best_score = -1
        best_newly_blocked = set()
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for x in candidates:
            # Compute new sums from adding x
            new_sums = set()
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            # Count how many currently-addable elements become blocked
            newly_blocked = set()
            for y in addable:
                if y == x:
                    continue
                # Check if y would be blocked after adding x
                blocked = False
                for a in selected:
                    if y + a in sums:
                        blocked = True
                        break
                if not blocked:
                    if 2 * y in sums:
                        blocked = True
                if not blocked:
                    # Check against new sums from x
                    if y + x in sums or y + x in new_sums:
                        blocked = True
                    elif 2 * y in new_sums:
                        blocked = True
                    else:
                        for a in selected:
                            if y + a in new_sums:
                                blocked = True
                                break
                if blocked:
                    newly_blocked.add(y)
            
            score = len(newly_blocked)
            if score > best_score:
                best_score = score
                best_x = x
                best_newly_blocked = newly_blocked
        
        if best_x is None:
            break
        
        # Add best_x
        sums.update([best_x + a for a in selected] + [2 * best_x])
        selected.append(best_x)
        sel_set.add(best_x)
        addable.discard(best_x)
        
        # Remove newly blocked elements
        addable -= best_newly_blocked
        
        # Recheck remaining addable elements
        still_addable = set()
        for x in addable:
            if _is_addable(x, selected, sums):
                still_addable.add(x)
        addable = still_addable
    
    return selected, sums

def _local_search_remove(selected, sums, rng, time_limit):
    """
    Try to remove elements from the set while maintaining maximality.
    Returns (shrunk_set, shrunk_sums) or None if time exceeded.
    """
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    
    improved = True
    while improved:
        if time.time() > time_limit:
            return None
        
        improved = False
        # Try removing each element
        for i in range(len(selected)):
            x = selected[i]
            # Remove x
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = sel_set - {x}
            
            # Check if still Sidon (it should be, since removing preserves Sidon)
            new_sums = _compute_sums(new_selected)
            
            # Check if still maximal
            if _is_maximal_fast(new_selected, new_sums):
                # Success! We can remove x
                selected = new_selected
                sums = new_sums
                sel_set = new_sel_set
                improved = True
                break
    
    return selected, sums

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    if not _is_sidon(selected):
        return False
    sums = _compute_sums(selected)
    return _is_maximal_fast(selected, sums)

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit
    
    # Get initial upper bound from simple greedy (ascending)
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)
    
    best_set = simple_selected
    best_size = len(best_set)
    
    # Verify it's maximal
    if not _verify_maximal(best_set):
        # Force maximality by adding all possible
        changed = True
        while changed:
            changed = False
            sums = _compute_sums(best_set)
            sel_set = set(best_set)
            for x in range(1, N + 1):
                if x not in sel_set:
                    if _is_addable(x, best_set, sums):
                        best_set.append(x)
                        changed = True
    
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Try multiple randomized greedy blocking strategies with local search
    for attempt in range(500):
        if time.time() > deadline - 1.0:
            break
        
        # Start with a random subset of small elements to seed
        seed_size = rng.randint(0, 4)
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
        
        result = _greedy_construct(seed_elements, rng, deadline - 0.5)
        if result is None:
            continue
        
        final_set, final_sums = result
        
        # Verify
        if not _verify_maximal(final_set):
            continue
        
        current_size = len(final_set)
        
        # Try local search to shrink
        if current_size < best_size or (current_size <= best_size and rng.random() < 0.3):
            shrunk_result = _local_search_remove(final_set, final_sums, rng, deadline - 0.3)
            if shrunk_result is not None:
                shrunk_set, shrunk_sums = shrunk_result
                if _verify_maximal(shrunk_set):
                    if len(shrunk_set) < best_size:
                        best_size = len(shrunk_set)
                        best_set = shrunk_set
                    current_size = len(shrunk_set)
        
        if current_size < best_size:
            best_size = current_size
            best_set = final_set
    
    return best_set
# EVOLVE-BLOCK-END
