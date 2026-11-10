"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _compute_sums(selected):
    """Compute all unordered pair sums (including diagonal) for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums

def _check_sidon(selected):
    """Check if selected is a valid Sidon set. Returns True if valid."""
    if len(selected) == 0:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]. Assumes valid Sidon."""
    selected_set = set(selected)
    sums = _compute_sums(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added without violating Sidon property
        can_add = True
        for a in selected:
            s1 = x + a
            if s1 in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        if can_add:
            return False
    return True

def _greedy_build(order, N):
    """Greedy construction: add elements in given order if they don't conflict."""
    selected = []
    sums = set()
    for x in order:
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

def _try_swap_remove(selected, N):
    """
    Try to reduce set size by removing one element and greedily rebuilding.
    Returns (new_set, success) where success=True if new_set is smaller and valid+maximal.
    """
    n = len(selected)
    if n <= 1:
        return None, False
    
    selected_set = set(selected)
    best_reduced = None
    
    # Try removing each element and greedily extending
    for i in range(n):
        reduced = [selected[j] for j in range(n) if j != i]
        if not reduced:
            continue
        if not _check_sidon(reduced):
            continue
        
        # Greedily extend to maximal
        reduced_set = set(reduced)
        ext = list(reduced)
        ext_sums = _compute_sums(reduced)
        
        for x in range(1, N + 1):
            if x in reduced_set or x in set(ext):
                continue
            added_sums = set()
            can_add = True
            for a in ext:
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
                ext.append(x)
                ext_sums.update(added_sums)
        
        if len(ext) < n and _check_sidon(ext) and _is_maximal(ext, N):
            if best_reduced is None or len(ext) < len(best_reduced):
                best_reduced = ext
    
    if best_reduced is not None:
        return best_reduced, True
    return None, False

def _try_swap_element(selected, N):
    """
    Try replacing one element with another to get a smaller or equal-size maximal set.
    Focus: find a swap that allows subsequent removal.
    Returns (new_set, success).
    """
    n = len(selected)
    if n <= 2:
        return None, False
    
    selected_set = set(selected)
    best_result = None
    
    # For each element to remove, try adding a different element
    for i in range(n):
        removed = selected[i]
        reduced = [selected[j] for j in range(n) if j != i]
        if not _check_sidon(reduced):
            continue
        
        reduced_set = set(reduced)
        reduced_sums = _compute_sums(reduced)
        
        # Find candidate replacements
        candidates = []
        for x in range(1, N + 1):
            if x in reduced_set or x == removed:
                continue
            # Check if x can be added to reduced
            can_add = True
            for a in reduced:
                s = x + a
                if s in reduced_sums:
                    can_add = False
                    break
            if can_add and 2 * x in reduced_sums:
                can_add = False
            if can_add:
                candidates.append(x)
        
        # Try each candidate replacement
        for x in candidates:
            new_set = reduced + [x]
            if len(new_set) != n:
                continue
            if not _check_sidon(new_set):
                continue
            # Now try to reduce this new set
            new_reduced, success = _try_swap_remove(new_set, N)
            if success and (best_result is None or len(new_reduced) < len(best_result)):
                best_result = new_reduced
    
    if best_result is not None:
        return best_result, True
    return None, False

def _annealing_search(start_set, N, deadline, rng):
    """
    Simulated annealing: iteratively try swap-remove moves with cooling.
    """
    current = list(start_set)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    temperature = 1.0
    cooling = 0.97
    min_temp = 0.01
    
    while time.time() < deadline and temperature > min_temp:
        # Try a random swap-remove move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try swap-remove on current
            new_set, success = _try_swap_remove(current, N)
            if success:
                if len(new_set) < current_size:
                    # Always accept improvement
                    current = new_set
                    current_size = len(current)
                    if current_size < best_size:
                        best = list(current)
                        best_size = current_size
                else:
                    # Accept with probability based on temperature
                    delta = len(new_set) - current_size
                    if rng.random() < min(1.0, max(0.0, temperature / max(1, delta))):
                        current = new_set
                        current_size = len(current)
        else:
            # Try swap-element then swap-remove
            new_set, success = _try_swap_element(current, N)
            if success:
                if len(new_set) < current_size:
                    current = new_set
                    current_size = len(current)
                    if current_size < best_size:
                        best = list(current)
                        best_size = current_size
                else:
                    delta = len(new_set) - current_size
                    if rng.random() < min(1.0, max(0.0, temperature / max(1, delta))):
                        current = new_set
                        current_size = len(current)
        
        temperature *= cooling
    
    return best

def construct():
    deadline = time.time() + 56.0  # 56s safety margin (60s limit)
    rng = random.Random(12345)
    
    # Phase 1: Generate diverse starting sets via random-order greedy
    best_set = None
    best_size = N + 1
    
    # Baseline: ascending order
    baseline = _greedy_build(list(range(1, N + 1)), N)
    if _check_sidon(baseline) and _is_maximal(baseline, N):
        best_size = len(baseline)
        best_set = list(baseline)
    
    # Multiple random orderings for diverse starts
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if _check_sidon(candidate) and _is_maximal(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)
                if best_size <= 16:
                    break
    
    if best_set is None:
        best_set = baseline
        best_size = len(best_set)
    
    # Phase 2: Simulated annealing from best set
    if time.time() < deadline - 5.0:
        anneal_deadline = deadline - 2.0
        improved = _annealing_search(best_set, N, anneal_deadline, rng)
        if improved is not None and len(improved) < best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Phase 3: Try annealing from a few other good starting sets
    if time.time() < deadline - 3.0:
        # Generate a few more diverse starts and anneal from them
        for attempt in range(30):
            if time.time() > deadline - 1.0:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            candidate = _greedy_build(order, N)
            if _check_sidon(candidate) and _is_maximal(candidate, N) and len(candidate) <= best_size:
                sub_deadline = min(deadline - 1.0, time.time() + 3.0)
                improved = _annealing_search(candidate, N, sub_deadline, rng)
                if improved is not None and len(improved) < best_size:
                    best_set = improved
                    best_size = len(best_set)
                    if best_size <= 16:
                        break
    
    return best_set
# EVOLVE-BLOCK-END
