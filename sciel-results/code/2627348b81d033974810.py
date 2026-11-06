"""Backtracking/greedy blocking-priority search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _compute_sums(selected):
    """Compute all pair sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to selected while preserving Sidon property."""
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_bitset(selected, sums):
    """
    Fast maximality check using precomputed conflict bitsets.
    For each element x, precompute which sums would conflict if x were added.
    Returns True if no missing element can be added.
    """
    sel_set = set(selected)
    K = len(selected)
    if K == 0:
        return False  # empty set is not maximal
    
    # Precompute: for each candidate x, check if adding x conflicts
    # A sum s conflicts with x if s = x + a for some a in selected, or s = 2*x
    # So x is addable iff: (x+a) not in sums for all a in selected, AND 2*x not in sums
    
    # Build a set of "blocked" candidates: x is blocked if exists a in selected s.t. x+a in sums, or 2*x in sums
    blocked = set()
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x is blocked
        is_blocked = False
        if 2 * x in sums:
            is_blocked = True
        else:
            for a in selected:
                if x + a in sums:
                    is_blocked = True
                    break
        if is_blocked:
            blocked.add(x)
    
    # If all missing elements are blocked, the set is maximal
    for x in range(1, N + 1):
        if x not in sel_set and x not in blocked:
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
                # y is already addable, so check if adding x blocks it
                if y + x in sums or y + x in new_sums:
                    newly_blocked.add(y)
                    continue
                if 2 * y in new_sums:
                    newly_blocked.add(y)
                    continue
                # Check if any y+a conflicts with new sums from x
                blocked = False
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
    
    return selected, sums

def _local_search_remove(selected, sums, rng, time_limit):
    """
    Try to remove elements from the set while maintaining maximality.
    Uses fast maximality check.
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
        # Try removing each element in random order for diversity
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            x = selected[i]
            
            # Compute sums that will be removed when x is removed
            removed_sums = set()
            for a in selected:
                if a >= x:
                    removed_sums.add(x + a)
            removed_sums.add(2 * x)
            
            # Tentatively remove
            new_sums = sums - removed_sums
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = sel_set - {x}
            
            # Check if still maximal using fast check
            if _is_maximal_bitset(new_selected, new_sums):
                # Commit removal
                selected = new_selected
                sums = new_sums
                sel_set = new_sel_set
                improved = True
                break
    
    return selected, sums

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    sums = _compute_sums(selected)
    # Check Sidon
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return _is_maximal_bitset(selected, sums)

def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # safety margin before 30s limit
    
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
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        
        # Start with a random subset to seed
        seed_size = rng.randint(0, 5)
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
        
        result = _greedy_construct(seed_elements, rng, deadline - 1.5)
        if result is None:
            continue
        
        final_set, final_sums = result
        
        # Verify
        if not _verify_maximal(final_set):
            continue
        
        current_size = len(final_set)
        
        # Try local search to shrink
        if current_size < best_size or (current_size <= best_size and rng.random() < 0.4):
            shrunk_result = _local_search_remove(final_set, final_sums, rng, deadline - 1.0)
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
