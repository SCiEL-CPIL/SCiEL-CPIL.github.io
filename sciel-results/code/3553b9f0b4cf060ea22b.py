"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 256
# EVOLVE-BLOCK-START

def _compute_sums(selected):
    """Compute all pairwise sums a+b with a<=b for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_sidon(selected, N):
    """Check if no missing element can be added without breaking Sidon property."""
    sel_set = set(selected)
    sums = _compute_sums(selected)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x breaks Sidon
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

def _build_conflict_graph(selected, N):
    """
    Build a conflict graph for the current Sidon set.
    Vertices: elements in [1, N] not in selected.
    Edge between x and y if x and y cannot both be added to selected
    because they would create a sum collision.
    
    Actually, for maximality, we need each missing element to be "blocked"
    by the existing set. The conflict graph approach: for each missing element x,
    determine which existing elements block it (i.e., x+a is in sums for some a,
    or 2x is in sums).
    """
    sel_set = set(selected)
    sums = _compute_sums(selected)
    
    # For each missing element, find what blocks it
    blockers = {}
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # x is blocked if 2x in sums OR x+a in sums for some a in selected
        block_reasons = []
        if 2 * x in sums:
            block_reasons.append(('self', x))  # blocked by 2x being a sum
        for a in selected:
            if x + a in sums:
                block_reasons.append(('pair', a))
        
        if not block_reasons:
            # x is NOT blocked, so set is not maximal
            return None, x, None
        
        blockers[x] = block_reasons
    
    return blockers, None, None

def _greedy_vertex_cover(blockers, selected, N, rng):
    """
    Use a heuristic minimum vertex cover approach to find a small set of elements
    whose removal would break maximality (i.e., make some missing element addable).
    
    Actually, we want to MINIMIZE the Sidon set while maintaining maximality.
    So we want to remove elements from `selected` such that the remaining set
    is still maximal.
    
    For each element a in selected, determine which missing elements it blocks.
    If we remove a, some missing elements might become unblocked.
    We need to ensure that for every missing element x, at least one of its
    blockers remains in the set.
    
    This is a hitting set / set cover problem:
    - Universe: missing elements that need to be blocked
    - Sets: for each a in selected, the set of missing elements that a blocks
    - Find minimum set of a's that covers all missing elements
    """
    sel_set = set(selected)
    sums = _compute_sums(selected)
    
    # For each missing element x, find all a in selected that block x
    # x is blocked by a if x+a is in sums (computed without a? No, with a)
    # Actually, x is blocked if 2x in sums OR x+a in sums for some a.
    # But x+a in sums means there exist b,c in selected with b+c = x+a.
    # This doesn't directly mean a blocks x.
    
    # Let me reconsider. The condition for x to be addable:
    # 1. 2x not in sums
    # 2. For all a in selected, x+a not in sums
    # 
    # So x is blocked if:
    # - 2x in sums, OR
    # - There exists a in selected such that x+a in sums
    #
    # The "blocking" is not directly attributable to individual elements a,
    # because x+a in sums means some pair (b,c) in selected sums to x+a.
    
    # Alternative approach: try removing elements one by one and check maximality.
    # This is what _shrink_maximal does, but let's make it smarter.
    
    # For each element a in selected, compute how many missing elements it "helps block"
    # by checking which missing elements would become unblocked if a were removed.
    
    missing = [x for x in range(1, N + 1) if x not in sel_set]
    
    # For each missing x, determine if it's currently blocked
    blocked_status = {}
    for x in missing:
        is_blocked = False
        if 2 * x in sums:
            is_blocked = True
        else:
            for a in selected:
                if x + a in sums:
                    is_blocked = True
                    break
        blocked_status[x] = is_blocked
    
    # If any missing element is not blocked, the set is not maximal
    if not all(blocked_status.values()):
        return selected  # Already not maximal, nothing to do
    
    # Now try to remove elements while maintaining maximality
    # Priority: remove elements that block the fewest missing elements
    # (i.e., their removal is least likely to break maximality)
    
    # For each a in selected, count how many missing elements x have x+a in sums
    # (excluding the case where the sum comes from a itself, i.e., a+a=2a)
    # Actually, x+a in sums means some pair (b,c) with b+c=x+a.
    # This is complex. Let's just try removals in order of "least critical".
    
    # Heuristic: remove elements in random order, check if still maximal
    result = list(selected)
    indices = list(range(len(result)))
    rng.shuffle(indices)
    
    for idx in indices:
        if not result:
            break
        candidate = result[:idx] + result[idx+1:]
        if _is_maximal_sidon(candidate, N):
            result = candidate
    
    return result

def _local_search_swap(selected, N, rng, deadline):
    """
    Try to improve the set by swapping elements: remove one, add another,
    then try to remove more.
    """
    sel_set = set(selected)
    missing = [x for x in range(1, N + 1) if x not in sel_set]
    
    if not missing:
        return selected
    
    best = selected
    best_size = len(selected)
    
    # Try swapping: remove a from selected, add x from missing
    for a in selected:
        if time.time() > deadline:
            break
        for x in missing:
            if time.time() > deadline:
                break
            candidate = [y for y in selected if y != a] + [x]
            if _is_sidon(candidate):
                # Now try to shrink
                shrunken = _greedy_vertex_cover(candidate, N, rng)
                if len(shrunken) < best_size:
                    best = shrunken
                    best_size = len(shrunken)
    
    return best

def _simulated_annealing(selected, N, rng, deadline):
    """
    Use simulated annealing with swap-remove neighborhood to find smaller
    maximal Sidon sets.
    """
    current = list(selected)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    temperature = 10.0
    cooling = 0.95
    
    while time.time() < deadline and temperature > 0.01:
        # Generate a neighbor: swap one element or remove one
        move_type = rng.random()
        
        if move_type < 0.7:
            # Swap: remove a, add x
            sel_set = set(current)
            missing = [x for x in range(1, N + 1) if x not in sel_set]
            if not missing:
                temperature *= cooling
                continue
            
            a = current[rng.randrange(len(current))]
            x = missing[rng.randrange(len(missing))]
            
            candidate = [y for y in current if y != a] + [x]
        else:
            # Remove one element
            if len(current) <= 16:
                temperature *= cooling
                continue
            a = current[rng.randrange(len(current))]
            candidate = [y for y in current if y != a]
        
        # Check if candidate is valid Sidon
        if not _is_sidon(candidate):
            temperature *= cooling
            continue
        
        # Check if candidate is maximal
        if not _is_maximal_sidon(candidate, N):
            # Try to fix by removing more elements? No, that makes it smaller but might break maximality.
            # Actually, if it's not maximal, we can't use it directly.
            # But we could try to add elements to make it maximal, but that increases size.
            # So skip this candidate.
            temperature *= cooling
            continue
        
        candidate_size = len(candidate)
        delta = candidate_size - current_size
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / temperature):
            current = candidate
            current_size = candidate_size
            
            if current_size < best_size:
                best = list(current)
                best_size = current_size
        
        temperature *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    num_restarts = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(i * 12345 + 67890)
        
        # Generate initial maximal set using greedy
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected = []
        sel_set = set()
        sums = set()
        
        for x in order:
            if 2 * x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if valid:
                selected.append(x)
                sel_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        if not selected:
            continue
        
        # Shrink using greedy removal
        result = _greedy_vertex_cover(selected, N, rng)
        
        # Local search with swaps
        if time.time() < deadline - 5:
            result = _local_search_swap(result, N, rng, deadline - 5)
        
        # Simulated annealing
        if time.time() < deadline - 2:
            result = _simulated_annealing(result, N, rng, deadline - 2)
        
        if len(result) < best_size:
            best = result
            best_size = len(result)
            
            if best_size <= 16:
                break
    
    if best is None:
        # Fallback
        order = list(range(1, N + 1))
        random.Random(0).shuffle(order)
        selected = []
        sel_set = set()
        sums = set()
        for x in order:
            if 2 * x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if valid:
                selected.append(x)
                sel_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected
    
    return best

# EVOLVE-BLOCK-END
