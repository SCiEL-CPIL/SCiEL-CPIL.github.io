import time
import random

def _check_and_add(values, sums, x):
    """Check if x can be appended to values. Returns (ok, new_tails)."""
    if values and x <= values[-1]:
        return False, None
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    if sums.intersection(tails):
        return False, None
    return True, tails

def _compute_all_sums(values):
    """Compute all consecutive block sums for a list."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _is_valid(values):
    """Check if values is strictly increasing and all consecutive block sums are distinct."""
    if not values:
        return True
    for i in range(len(values) - 1):
        if values[i] >= values[i+1]:
            return False
    sums = _compute_all_sums(values)
    return len(sums) == len(values) * (len(values) + 1) // 2

def _try_insert(values, sums, x, pos):
    """Try to insert x at position pos in values. Returns (ok, new_values, new_sums)."""
    if pos < 0 or pos > len(values):
        return False, None, None
    # Check strictly increasing
    if pos > 0 and x <= values[pos-1]:
        return False, None, None
    if pos < len(values) and x >= values[pos]:
        return False, None, None
    
    new_values = values[:pos] + [x] + values[pos:]
    new_sums = _compute_all_sums(new_values)
    if len(new_sums) != len(new_values) * (len(new_values) + 1) // 2:
        return False, None, None
    return True, new_values, new_sums

def _greedy_order(order, values_init=None, sums_init=None):
    """Run greedy with a given order of candidates."""
    if values_init is None:
        values = []
        sums = set()
    else:
        values = list(values_init)
        sums = set(sums_init)
    
    for x in order:
        ok, tails = _check_and_add(values, sums, x)
        if ok:
            values.append(x)
            sums.update(tails)
    return values, sums

def _gap_fill(values, sums, max_iterations=5000):
    """Try to insert unused numbers into gaps in the sequence."""
    current_values = list(values)
    current_sums = set(sums)
    used = set(current_values)
    
    for _ in range(max_iterations):
        improved = False
        # Try inserting each unused number at each possible position
        unused = [x for x in range(1, 257) if x not in used]
        random.shuffle(unused)
        
        for x in unused:
            best_pos = -1
            best_new_values = None
            best_new_sums = None
            
            for pos in range(len(current_values) + 1):
                ok, nv, ns = _try_insert(current_values, current_sums, x, pos)
                if ok:
                    # Prefer positions that maximize future flexibility
                    # (e.g., middle positions)
                    best_pos = pos
                    best_new_values = nv
                    best_new_sums = ns
                    break  # Take first valid position for speed
            
            if best_new_values is not None:
                current_values = best_new_values
                current_sums = best_new_sums
                used.add(x)
                improved = True
        
        if not improved:
            break
    
    return current_values, current_sums

def _try_local_improve(values, sums, max_iterations=5000):
    """Try to improve by removing an element and re-inserting elsewhere."""
    best_len = len(values)
    current_values = list(values)
    current_sums = set(sums)
    
    for _ in range(max_iterations):
        # Try removing a random element
        idx = random.randint(0, len(current_values) - 1)
        removed = current_values[idx]
        
        # Remove it and rebuild sums
        new_values = current_values[:idx] + current_values[idx+1:]
        # Rebuild sums for new_values
        new_sums = _compute_all_sums(new_values)
        
        # Try re-inserting removed at various positions
        best_pos = -1
        best_new_values = None
        best_new_sums = None
        for pos in range(len(new_values) + 1):
            ok, nv, ns = _try_insert(new_values, new_sums, removed, pos)
            if ok:
                if len(nv) > best_len or (best_new_values is None):
                    best_pos = pos
                    best_new_values = nv
                    best_new_sums = ns
        
        if best_new_values is not None and len(best_new_values) >= len(current_values):
            current_values = best_new_values
            current_sums = best_new_sums
            if len(current_values) > best_len:
                best_len = len(current_values)
    
    return current_values, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Phase 1: Natural order greedy
    values, sums = _greedy_order(range(1, 257))
    best_values = list(values)
    best_sums = set(sums)
    best_len = len(best_values)
    
    # Phase 2: Randomized greedy restarts
    rng = random.Random(42)
    candidates = list(range(1, 257))
    
    while time.time() < deadline - 10:
        rng.shuffle(candidates)
        vals, sums = _greedy_order(candidates)
        if len(vals) > best_len:
            best_len = len(vals)
            best_values = vals
            best_sums = sums
    
    # Phase 3: Gap filling on best
    if time.time() < deadline - 5:
        filled_vals, filled_sums = _gap_fill(best_values, best_sums, max_iterations=100)
        if len(filled_vals) > best_len:
            best_len = len(filled_vals)
            best_values = filled_vals
            best_sums = filled_sums
    
    # Phase 4: Local improvement on best
    if time.time() < deadline - 2:
        improved_vals, improved_sums = _try_local_improve(best_values, best_sums, max_iterations=2000)
        if len(improved_vals) > best_len:
            best_len = len(improved_vals)
            best_values = improved_vals
            best_sums = improved_sums
    
    return best_values
