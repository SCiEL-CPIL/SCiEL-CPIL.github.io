"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_sidon_fast(selected):
    """Fast Sidon check using numpy."""
    arr = np.array(selected, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return True
    total = n * (n + 1) // 2
    sums = np.zeros(total, dtype=np.int32)
    idx = 0
    for i in range(n):
        sums[idx] = 2 * arr[i]
        idx += 1
        for j in range(i + 1, n):
            sums[idx] = arr[i] + arr[j]
            idx += 1
    return len(np.unique(sums)) == total

def _compute_sums_set(selected):
    """Compute all pairwise sums a+b for a<=b in selected as a set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_maximal_fast(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    sums = _compute_sums_set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # x can be added iff 2x not in sums AND x+s not in sums for all s
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_construct(order, initial=None):
    """Greedy Sidon set construction following the given order, optionally starting from initial."""
    if initial is not None:
        selected = list(initial)
        sums = _compute_sums_set(selected)
    else:
        selected = []
        sums = set()
    selected_set = set(selected)
    for x in order:
        if x in selected_set:
            continue
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            continue
        if sums.intersection(new_sums):
            continue
        selected.append(x)
        sums.update(new_sums)
        selected_set.add(x)
    return selected

def _forced_additions(core):
    """
    Given a Sidon core, greedily add only elements that are forced for maximality.
    An element x is 'forced' if its absence would allow some other missing element to be added.
    We iterate: find all missing elements that can be added; if any exist, we must add
    elements to block them. We add the element that blocks the most addable elements.
    """
    selected = list(core)
    sums = _compute_sums_set(selected)
    selected_set = set(selected)
    
    def _addable_elements():
        """Return list of missing elements that can be added to selected."""
        addable = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if 2 * x in sums:
                continue
            conflict = False
            for s in selected:
                if x + s in sums:
                    conflict = True
                    break
            if not conflict:
                addable.append(x)
        return addable
    
    max_iter = 1000
    for _ in range(max_iter):
        addable = _addable_elements()
        if not addable:
            break
        
        # For each candidate element x (not in selected), count how many addable elements it blocks
        # x blocks y if: 2y in sums_after_adding_x OR y+s in sums_after_adding_x for some s
        # Simpler: try adding each candidate and see how many addable elements become non-addable
        best_candidate = None
        best_blocked = -1
        
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        for cand in candidates:
            # Check if cand can be added (must be Sidon-compatible)
            new_sums = [cand + a for a in selected] + [2 * cand]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(new_sums):
                continue
            
            # Compute how many addable elements are blocked by adding cand
            new_sums_set = sums | set(new_sums)
            new_selected = selected + [cand]
            blocked = 0
            for y in addable:
                if y == cand:
                    continue
                # Check if y is still addable after adding cand
                if 2 * y in new_sums_set:
                    blocked += 1
                    continue
                y_conflict = False
                for s in new_selected:
                    if y + s in new_sums_set:
                        y_conflict = True
                        break
                if y_conflict:
                    blocked += 1
            
            if blocked > best_blocked:
                best_blocked = blocked
                best_candidate = cand
        
        if best_candidate is None:
            # No candidate can be added; this shouldn't happen if addable is non-empty
            break
        
        # Add the best candidate
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        sums = _compute_sums_set(selected)
    
    return selected

def _local_search_reduce(selected, deadline):
    """
    Try to remove elements while maintaining maximality.
    Multiple passes with swap moves to escape local minima.
    """
    current = list(selected)
    improved = True
    pass_count = 0
    max_passes = 50
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Phase 1: Try removals
        for x in list(current):
            if time.time() > deadline:
                return current
            new_list = [y for y in current if y != x]
            if len(new_list) == 0:
                continue
            # Subset of Sidon is Sidon, so just check maximality
            if _is_maximal_fast(new_list):
                current = new_list
                improved = True
                break  # Restart after removal
        
        if improved:
            continue
        
        # Phase 2: Try swaps (remove x, add y) to escape local minima
        current_set = set(current)
        for x in list(current):
            if time.time() > deadline:
                return current
            without_x = [y for y in current if y != x]
            if len(without_x) == 0:
                continue
            without_x_set = set(without_x)
            sums_x = _compute_sums_set(without_x)
            
            # Try adding each missing element to without_x
            for y in range(1, N + 1):
                if y in without_x_set:
                    continue
                # Check if y can be added to without_x
                new_sums = [y + a for a in without_x] + [2 * y]
                if len(set(new_sums)) != len(new_sums):
                    continue
                if sums_x.intersection(new_sums):
                    continue
                
                candidate = without_x + [y]
                if _is_maximal_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def _construct_spread_core(seed_val):
    """Construct a well-spread initial core using modular/pattern-based selection."""
    rng = random.Random(seed_val)
    core = []
    
    # Strategy: pick elements that are spread out and likely Sidon-compatible
    # Use a greedy approach with a bias toward spread
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # First, try to build a small core with good spread
    # Sort by "spread" score: prefer elements far from existing ones
    for x in candidates:
        if len(core) >= 15:  # Keep core small
            break
        # Check if x can be added to core
        new_sums = [x + a for a in core] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            continue
        core_sums = _compute_sums_set(core)
        if core_sums.intersection(new_sums):
            continue
        core.append(x)
    
    return core

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    if len(selected) < best_size:
        best = selected
        best_size = len(selected)
    
    # Local search to reduce size
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
    
    # Strategy 2: Direct minimization - build small core, then forced additions
    if time.time() < deadline - 5.0:
        for trial in range(200):
            if time.time() > deadline:
                break
            core = _construct_spread_core(trial)
            if len(core) == 0:
                continue
            if not _is_sidon_fast(core):
                continue
            # Add forced elements for maximality
            result = _forced_additions(core)
            if not _is_sidon_fast(result):
                continue
            if not _is_maximal_fast(result):
                continue
            if len(result) < best_size:
                best = result
                best_size = len(result)
                # Try local search on this better result
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 3: Try many different random orders with greedy
    if time.time() < deadline - 3.0:
        for trial in range(150):
            if time.time() > deadline:
                break
            rng = random.Random(trial * 7919 + 12345)
            order = list(range(1, N + 1))
            rng.shuffle(order)
            selected = _greedy_construct(order)
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 4: Start with different initial sizes and build up
    if time.time() < deadline - 5.0:
        for init_size in range(2, 10):
            if time.time() > deadline:
                break
            for trial in range(15):
                if time.time() > deadline:
                    break
                rng = random.Random(trial * 7919 + init_size * 104729)
                initial = rng.sample(range(1, N + 1), min(init_size, N))
                initial.sort()
                
                if not _is_sidon_fast(initial):
                    continue
                
                remaining = [x for x in range(1, N + 1) if x not in set(initial)]
                rng.shuffle(remaining)
                
                selected = _greedy_construct(remaining, initial=initial)
                if len(selected) < best_size:
                    best = selected
                    best_size = len(selected)
                    if time.time() < deadline - 2.0:
                        reduced = _local_search_reduce(best, deadline)
                        if len(reduced) < best_size:
                            best = reduced
                            best_size = len(reduced)
    
    # Final safety check
    if best is None:
        # Fallback: ascending greedy
        best = _greedy_construct(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
