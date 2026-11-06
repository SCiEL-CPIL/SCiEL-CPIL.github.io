"""Combined program: Bitset-accelerated search for small maximal Sidon sets in [1,256].

Strategy:
1. Bitset-based sum tracking for O(1) conflict checks.
2. Diverse greedy constructions (ascending, descending, random, restrictive).
3. Smart reduction: remove element whose absence unblocks fewest new elements.
4. Swap-based local search to escape local optima.
5. 58s deadline with safety margin.
"""

import time
import random

N = 256
MAX_SUM = 2 * N  # 512

def _is_sidon_bitset(selected):
    """Check if selected is a Sidon set using bitset."""
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if (sums >> (2 * a)) & 1:
            return False
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            if (sums >> s) & 1:
                return False
            sums |= (1 << s)
    return True

def _compute_sums_bitset(selected):
    """Compute the bitset of all pair sums a+b for a<=b in selected."""
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            sums |= (1 << s)
    return sums

def _is_maximal_bitset(selected, sums_mask):
    """Check if selected is inclusion-maximal using bitset sums."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check diagonal: 2x must be in sums
        if (sums_mask >> (2 * x)) & 1:
            continue
        # Check off-diagonal: for all a in selected, x+a must be in sums
        blocked = False
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _count_newly_addable(selected_without_x, sums_without_x, original_selected):
    """Count how many elements become addable when we remove x from the set."""
    sel_set = set(selected_without_x)
    count = 0
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added to selected_without_x
        if (sums_without_x >> (2 * x)) & 1:
            continue
        ok = True
        for a in selected_without_x:
            if (sums_without_x >> (x + a)) & 1:
                ok = False
                break
        if ok:
            count += 1
    return count

def _greedy_bitset(order, deadline):
    """Greedy construction using bitset for fast checks."""
    selected = []
    sums = 0
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        if (sums >> (2 * x)) & 1:
            continue
        ok = True
        for a in selected:
            if (sums >> (x + a)) & 1:
                ok = False
                break
        if ok:
            selected.append(x)
            sums |= (1 << (2 * x))
            for a in selected[:-1]:
                sums |= (1 << (x + a))
    return selected

def _greedy_ascending():
    """Greedy: pick smallest possible element."""
    return _greedy_bitset(range(1, N + 1), time.time() + 100)

def _greedy_descending():
    """Greedy: pick largest possible element."""
    return _greedy_bitset(range(N, 0, -1), time.time() + 100)

def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_bitset(order, time.time() + 100)

def _greedy_most_restrictive(deadline, rng=None):
    """
    Greedy that picks the most restrictive element at each step.
    For each candidate x, compute how many other elements would still be addable.
    Pick x that leaves the fewest options.
    """
    available = list(range(1, N + 1))
    if rng:
        rng.shuffle(available)
    selected = []
    sums = 0
    
    while available and time.time() < deadline:
        best_x = None
        best_options = float('inf')
        best_idx = -1
        
        for i, x in enumerate(available):
            # Check if x can be added
            if (sums >> (2 * x)) & 1:
                continue
            ok = True
            for a in selected:
                if (sums >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Count how many remaining elements could still be added after x
            temp_sums = sums | (1 << (2 * x))
            for a in selected:
                temp_sums |= (1 << (x + a))
            
            options = 0
            for y in available:
                if y == x:
                    continue
                if (temp_sums >> (2 * y)) & 1:
                    continue
                ok_y = True
                for a in selected:
                    if (temp_sums >> (y + a)) & 1:
                        ok_y = False
                        break
                if not ok_y:
                    continue
                # Also check y+x
                if (temp_sums >> (y + x)) & 1:
                    continue
                options += 1
            
            if options < best_options:
                best_options = options
                best_x = x
                best_idx = i
        
        if best_x is None:
            break
        
        selected.append(best_x)
        available.pop(best_idx)
        sums |= (1 << (2 * best_x))
        for a in selected[:-1]:
            sums |= (1 << (best_x + a))
    
    return selected

def _reduce_smart(start_set, deadline, rng):
    """
    Smart reduction: iteratively remove the element whose absence
    causes the fewest new elements to become addable.
    This is a heuristic for minimum vertex cover on the dependency graph.
    """
    current = list(start_set)
    if len(current) <= 1:
        return current
    
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_remove_idx = -1
        best_newly_addable = float('inf')
        
        # Try removing each element
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            x = current[i]
            new_current = current[:i] + current[i+1:]
            if not new_current:
                continue
            
            new_sums = _compute_sums_bitset(new_current)
            
            # Check maximality
            if not _is_maximal_bitset(new_current, new_sums):
                continue
            
            # Count newly addable elements
            newly = _count_newly_addable(new_current, new_sums, current)
            
            if newly < best_newly_addable:
                best_newly_addable = newly
                best_remove_idx = i
        
        if best_remove_idx >= 0:
            current = current[:best_remove_idx] + current[best_remove_idx+1:]
            sums_mask = _compute_sums_bitset(current)
            improved = True
    
    return current

def _reduce_random(start_set, deadline, rng):
    """
    Random reduction: try removing elements in random order.
    Accept removal if set remains maximal.
    """
    current = list(start_set)
    if len(current) <= 1:
        return current
    
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            x = current[idx]
            new_current = current[:idx] + current[idx+1:]
            if not new_current:
                continue
            
            new_sums = _compute_sums_bitset(new_current)
            
            if _is_maximal_bitset(new_current, new_sums):
                current = new_current
                sums_mask = new_sums
                improved = True
                break
    
    return current

def _swap_improve(start_set, deadline, rng):
    """
    Try swaps to find better configurations that allow further reduction.
    """
    current = list(start_set)
    current_set = set(current)
    
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 50:
        improved = False
        iters += 1
        
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N+1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            temp_current = [x for x in current if x != r]
            temp_set = set(temp_current)
            
            for a in add_candidates:
                if time.time() > deadline:
                    break
                if a in temp_set:
                    continue
                
                candidate = temp_current + [a]
                candidate.sort()
                
                if not _is_sidon_bitset(candidate):
                    continue
                
                cand_sums = _compute_sums_bitset(candidate)
                if _is_maximal_bitset(candidate, cand_sums):
                    # Accept swap
                    current = candidate
                    current_set = set(candidate)
                    improved = True
                    break
            if improved:
                break
    return current

def _make_maximal(current, deadline):
    """
    Ensure the set is maximal by adding any missing element that doesn't break Sidon.
    """
    current = list(current)
    current_set = set(current)
    sums_mask = _compute_sums_bitset(current)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Check if adding x keeps it Sidon
            if (sums_mask >> (2 * x)) & 1:
                continue
            ok = True
            for a in current:
                if (sums_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Add it
            current.append(x)
            current_set.add(x)
            sums_mask |= (1 << (2 * x))
            for a in current[:-1]:
                sums_mask |= (1 << (x + a))
            changed = True
            break
            
    return current

def construct():
    start_time = time.time()
    # Use 58s deadline to be safe against 60s hard limit
    deadline = start_time + 58.0
    
    # Generate diverse starting sets
    candidates = []
    
    # 1. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 2. Greedy descending
    candidates.append(_greedy_descending())
    
    # 3. Most restrictive greedy
    candidates.append(_greedy_most_restrictive(deadline))
    
    # 4. Multiple random greedy - more trials with bitset speed
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        candidates.append(_greedy_random(seed))
    
    # 5. Most restrictive with different seeds
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        candidates.append(_greedy_most_restrictive(deadline, rng))
    
    # Filter valid Sidon sets
    valid_candidates = []
    for c in candidates:
        if c and _is_sidon_bitset(c):
            valid_candidates.append(c)
            
    if not valid_candidates:
        return _greedy_ascending()
        
    # Sort by size to prioritize smaller sets
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Apply local search to top candidates
    rng = random.Random(42)
    
    # Process top candidates with both reduction strategies
    for c in valid_candidates[:60]:
        if time.time() > deadline - 2:
            break
        
        # Try smart reduction first
        reduced = _reduce_smart(c, deadline - 1, rng)
        if _is_sidon_bitset(reduced) and _is_maximal_bitset(reduced, _compute_sums_bitset(reduced)):
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced
        
        # Try random reduction if time permits
        if time.time() < deadline - 3:
            reduced2 = _reduce_random(c, deadline - 1, rng)
            if _is_sidon_bitset(reduced2) and _is_maximal_bitset(reduced2, _compute_sums_bitset(reduced2)):
                if len(reduced2) < best_size:
                    best_size = len(reduced2)
                    best_set = reduced2
        
        # Try swaps on the best found so far
        if time.time() < deadline - 5 and len(best_set) > 8:
            swapped = _swap_improve(best_set, deadline - 2, rng)
            if _is_sidon_bitset(swapped) and _is_maximal_bitset(swapped, _compute_sums_bitset(swapped)):
                if len(swapped) < best_size:
                    best_size = len(swapped)
                    best_set = swapped
    
    # Final verification
    if not _is_sidon_bitset(best_set):
        return _greedy_ascending()
        
    # Ensure maximality (safety check)
    sums = _compute_sums_bitset(best_set)
    if not _is_maximal_bitset(best_set, sums):
        best_set = _make_maximal(best_set, deadline)
        # Re-verify
        if not _is_sidon_bitset(best_set):
            return _greedy_ascending()
        
    return best_set
