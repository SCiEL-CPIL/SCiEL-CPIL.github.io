"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_maximal_sidon_fast(selected, N):
    """Check if no missing element can be added without breaking Sidon property."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            sums.add(s)
    
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
            return False
    return True

def _greedy_blocking_maximal_fast(rng, N):
    """
    Generate a maximal Sidon set by greedily adding the element
    that blocks the most other unblocked elements.
    Optimized: compute blocking impact incrementally.
    """
    selected = []
    sel_set = set()
    sums = set()
    
    while True:
        # Compute blocked set
        blocked = set()
        for s in sums:
            if s % 2 == 0:
                xc = s // 2
                if 1 <= xc <= N:
                    blocked.add(xc)
            for a in selected:
                xc = s - a
                if 1 <= xc <= N:
                    blocked.add(xc)
        
        # Find unblocked
        unblocked = [x for x in range(1, N + 1) if x not in sel_set and x not in blocked]
        
        if not unblocked:
            break
        
        # Choose x that maximizes number of newly blocked unblocked elements
        best_x = None
        best_score = -1
        
        unblocked_set = set(unblocked)
        
        for x in unblocked:
            # New sums when adding x: 2x, x+a for a in selected
            # An unblocked element y (y != x) becomes blocked if:
            #   2y in new_sums, or y+a in new_sums for some a in selected ∪ {x}
            # new_sums = sums ∪ {2x} ∪ {x+a : a in selected}
            # y is already unblocked, so 2y not in sums, and y+a not in sums for all a in selected.
            # So y becomes blocked iff:
            #   2y == 2x (impossible since y != x), or
            #   2y == x+a for some a in selected, or
            #   y+a == 2x for some a in selected, or
            #   y+a == x+b for some a,b in selected (but this was already checked)
            
            # More carefully:
            # y is blocked by new sums if there exists a sum s in new_sums such that
            # 2y = s or y + a' = s for some a' in selected ∪ {x}
            
            # Since y was unblocked: 2y not in sums, and y+a not in sums for all a in selected.
            # New sums are: 2x, x+a for a in selected.
            
            # Check 2y == 2x: y == x, excluded.
            # Check 2y == x + a for a in selected: y = (x+a)/2, must be integer
            # Check y + a' == 2x for a' in selected ∪ {x}:
            #   a' = x: y + x == 2x => y == x, excluded
            #   a' in selected: y == 2x - a'
            # Check y + a' == x + b for a', b in selected:
            #   This means y = x + b - a'. Since y was unblocked, y+a not in sums for all a in selected.
            #   x+b is in new_sums. y + a' = x + b => y = x + b - a'.
            #   For this to block y, we need y + a' in new_sums, i.e., x + b in new_sums (yes) and a' in selected ∪ {x}.
            #   But we also need that this particular sum wasn't already blocking y. Since y was unblocked,
            #   y + a' was not in sums for any a' in selected. Now y + a' = x + b which is in new_sums.
            #   So yes, this blocks y.
            
            # Simplified: y becomes blocked if:
            # 1. 2y == x + a for some a in selected (i.e., x + a is even and y = (x+a)//2)
            # 2. y + a == 2x for some a in selected (i.e., y = 2x - a)
            # 3. y + a == x + b for some a, b in selected (i.e., y = x + b - a)
            
            count = 0
            # Precompute for speed
            sel_list = selected
            sx = 2 * x
            
            for y in unblocked:
                if y == x:
                    continue
                is_blocked = False
                
                # Check condition 1: 2y == x + a for some a in selected
                two_y = 2 * y
                for a in sel_list:
                    if x + a == two_y:
                        is_blocked = True
                        break
                if is_blocked:
                    count += 1
                    continue
                    
                # Check condition 2: y + a == 2x for some a in selected
                for a in sel_list:
                    if y + a == sx:
                        is_blocked = True
                        break
                if is_blocked:
                    count += 1
                    continue
                    
                # Check condition 3: y + a == x + b for some a, b in selected
                # Equivalent: y = x + b - a for some a, b in selected
                # Or: y - x = b - a for some a, b in selected
                # Precompute differences b - a for a, b in selected
                # But this is O(S²) per y, too slow.
                # Alternative: for each a in selected, check if y + a - x is in selected
                for a in sel_list:
                    b_candidate = y + a - x
                    if b_candidate in sel_set:
                        is_blocked = True
                        break
                if is_blocked:
                    count += 1
                    
            if count > best_score:
                best_score = count
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        selected.append(best_x)
        sel_set.add(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
            
    return selected

def _shrink_maximal_fast(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = selected[:idx] + selected[idx+1:]
            
            if _is_maximal_sidon_fast(candidate, N):
                selected = candidate
                improved = True
                break
        
        if not improved:
            break
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    for i in range(2000):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 7919 + 104729)
        
        sel = _greedy_blocking_maximal_fast(rng, N)
        
        if not sel:
            continue
            
        sel = _shrink_maximal_fast(sel, N, deadline)
        
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            if best_size <= 16:
                break

    if best is None:
        sel = _greedy_blocking_maximal_fast(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
