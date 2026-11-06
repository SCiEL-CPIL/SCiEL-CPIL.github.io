import random
import time
import numpy as np

N = 256

def _compute_sums(selected):
    """Compute all pairwise sums (a+b for a<=b)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all pair sums distinct)."""
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
    """Check if selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # x can be added if 2x not in sums AND x+a not in sums for all a in selected
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

def _get_unblocked(selected, sums):
    """Get list of missing elements that could be added (unblocked)."""
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
    """Try to add x to selected. Returns (new_selected, new_sums) or (None, None)."""
    if 2 * x in sums:
        return None, None
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        s = x + a
        if s in new_sums:
            return None, None
        new_sums.add(s)
    return selected + [x], new_sums

def _greedy_maximize(selected, sums, deadline):
    """Greedily add elements until maximal, choosing element that blocks most others."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            new_sel, new_sums = _try_add(selected, sums, x)
            if new_sel is None:
                continue
            
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in new_sel:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        new_sel, new_sums = _try_add(selected, sums, best_x)
        if new_sel is None:
            break
        selected = new_sel
        sums = new_sums
    
    return selected, sums

def _local_search_reduce(selected, deadline):
    """Try to remove elements while maintaining Sidon + maximality."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            if _is_valid_sidon(candidate):
                sums = _compute_sums(candidate)
                if _is_maximal(candidate, sums):
                    selected = candidate
                    improved = True
                    break
    return selected

def _swap_search(selected, sums, deadline):
    """Try swapping one element for another to reduce size or improve blocking."""
    selected = list(selected)
    sums = set(sums)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sel_set = set(selected)
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                # Remove selected[i], add x
                temp = selected[:i] + selected[i+1:]
                new_sel, new_sums = _try_add(temp, _compute_sums(temp), x)
                if new_sel is None:
                    continue
                if len(new_sel) < len(selected):
                    # This would be a reduction, but we're swapping so same size
                    pass
                # Check if new set is maximal
                if _is_maximal(new_sel, new_sums):
                    # Prefer swaps that lead to smaller sets after reduction
                    selected = new_sel
                    sums = new_sums
                    sel_set = set(selected)
                    improved = True
                    break
            if improved:
                break
    return selected, sums

def _hitting_set_reduce(selected, sums, deadline):
    """
    Use a hitting-set perspective: for each missing element m, identify which
    selected elements are 'critical' for blocking m. Remove non-critical elements.
    
    An element a is critical for blocking m if removing a would make m addable.
    """
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sel_set = set(selected)
        
        # For each selected element, count how many missing elements it uniquely blocks
        unique_block = {}
        for a in selected:
            count = 0
            for m in range(1, N + 1):
                if m in sel_set:
                    continue
                # Check if m is currently blocked
                if 2 * m in sums:
                    continue
                m_blocked = False
                for b in selected:
                    if m + b in sums:
                        m_blocked = True
                        break
                if not m_blocked:
                    continue
                
                # m is blocked. Check if removing a would unblock m.
                # After removing a, we need to recompute sums without a.
                # But that's expensive. Instead, check if a is the only element
                # that blocks m via sums.
                # m is blocked if 2m in sums OR m+b in sums for some b.
                # If 2m in sums, a might contribute to that sum.
                # If m+b in sums, a might be b or contribute to that sum.
                
                # Simplified: check if a is needed for any blocking sum
                # 2m in sums: a contributes if a+a=2m (a=m, but m not in sel) or a+b=2m
                # m+b in sums: a contributes if a=m (no) or a is the b, or a+c=m+b
                
                # This is complex. Use a simpler heuristic:
                # Try removing a and check if m becomes unblocked.
                # But only do this for a few candidates to save time.
                
                # Actually, let's just do the full check but limit to a few elements.
                pass
            
            unique_block[a] = count
        
        # Try removing elements that seem least critical
        # Sort by some heuristic and try removing
        for a in selected:
            if time.time() > deadline:
                break
            candidate = [x for x in selected if x != a]
            if not candidate:
                continue
            if _is_valid_sidon(candidate):
                new_sums = _compute_sums(candidate)
                if _is_maximal(candidate, new_sums):
                    selected = candidate
                    sums = new_sums
                    sel_set = set(selected)
                    improved = True
                    break
    
    return selected, sums

def _random_sidon_init(rng, target_size):
    """Generate a random Sidon set of approximately target_size elements."""
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(selected) >= target_size:
            break
        new_sel, new_sums = _try_add(selected, sums, x)
        if new_sel is not None:
            selected = new_sel
            sums = new_sums
    
    return selected, sums

def _structured_init(base, step):
    """Generate a structured Sidon set."""
    selected = []
    sums = set()
    for x in range(base, N + 1, step):
        new_sel, new_sums = _try_add(selected, sums, x)
        if new_sel is not None:
            selected = new_sel
            sums = new_sums
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Random initial Sidon set + greedy maximize + local search
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        target_size = rng.randint(8, 22)
        selected, sums = _random_sidon_init(rng, target_size)
        
        if not selected:
            continue
        
        selected, sums = _greedy_maximize(selected, sums, deadline)
        
        if _is_valid_sidon(selected) and _is_maximal(selected, sums):
            if len(selected) < best_size:
                best_size = len(selected)
                best = list(selected)
            
            if time.time() < deadline:
                # Local search reduction
                reduced = _local_search_reduce(selected, deadline)
                if _is_valid_sidon(reduced):
                    red_sums = _compute_sums(reduced)
                    if _is_maximal(reduced, red_sums):
                        if len(reduced) < best_size:
                            best_size = len(reduced)
                            best = list(reduced)
                
                # Hitting-set reduction
                if time.time() < deadline:
                    hs_result, hs_sums = _hitting_set_reduce(reduced, red_sums, deadline)
                    if _is_valid_sidon(hs_result) and _is_maximal(hs_result, hs_sums):
                        if len(hs_result) < best_size:
                            best_size = len(hs_result)
                            best = list(hs_result)
        
        if best_size <= 13:
            break
    
    # Strategy 2: Structured initial sets
    if time.time() < deadline:
        for base in range(2, 15):
            if time.time() > deadline:
                break
            for step in range(base + 1, base * 3 + 1):
                if time.time() > deadline:
                    break
                selected, sums = _structured_init(base, step)
                
                if not selected:
                    continue
                
                selected, sums = _greedy_maximize(selected, sums, deadline)
                
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                    
                    if time.time() < deadline:
                        reduced = _local_search_reduce(selected, deadline)
                        if _is_valid_sidon(reduced):
                            red_sums = _compute_sums(reduced)
                            if _is_maximal(reduced, red_sums):
                                if len(reduced) < best_size:
                                    best_size = len(reduced)
                                    best = list(reduced)
    
    # Strategy 3: More random restarts with different seeds
    if time.time() < deadline:
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            target_size = rng.randint(6, 18)
            selected, sums = _random_sidon_init(rng, target_size)
            
            if not selected:
                continue
            
            selected, sums = _greedy_maximize(selected, sums, deadline)
            
            if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                if len(selected) < best_size:
                    best_size = len(selected)
                    best = list(selected)
                
                if time.time() < deadline:
                    reduced = _local_search_reduce(selected, deadline)
                    if _is_valid_sidon(reduced):
                        red_sums = _compute_sums(reduced)
                        if _is_maximal(reduced, red_sums):
                            if len(reduced) < best_size:
                                best_size = len(reduced)
                                best = list(reduced)
    
    # Fallback: simple greedy
    if best is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            new_sel, new_sums = _try_add(selected, sums, x)
            if new_sel is not None:
                selected = new_sel
                sums = new_sums
        return selected
    
    return best
