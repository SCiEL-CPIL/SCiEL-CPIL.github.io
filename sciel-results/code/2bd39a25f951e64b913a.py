"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_sidon(selected, N):
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
        # Check if adding x breaks Sidon
        # x can be added if 2x not in sums and x+a not in sums for all a in selected
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
    Optimized using difference set membership.
    """
    selected = []
    sel_set = set()
    sums = set()
    diffs = set()  # a - b for a, b in selected

    while True:
        # Find unblocked elements
        unblocked = []
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if 2 * y in sums:
                continue
            blocked = False
            for a in selected:
                if y + a in sums:
                    blocked = True
                    break
            if not blocked:
                unblocked.append(y)

        if not unblocked:
            break

        # Choose x that maximizes blocking of other unblocked elements
        best_x = None
        best_count = -1

        for x in unblocked:
            count = 0
            for y in unblocked:
                if y == x:
                    continue
                # y is blocked by x if:
                # 1. 2y - x in sel_set  (2y = x + a => a = 2y - x)
                # 2. 2x - y in sel_set  (y + a = 2x => a = 2x - y)
                # 3. y - x in diffs     (y + a = x + a' => a' - a = y - x)
                
                if (2 * y - x) in sel_set:
                    count += 1
                elif (2 * x - y) in sel_set:
                    count += 1
                elif (y - x) in diffs:
                    count += 1

            if count > best_count:
                best_count = count
                best_x = x

        if best_x is None:
            break

        # Add best_x
        selected.append(best_x)
        sel_set.add(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
            diffs.add(best_x - a)
            diffs.add(a - best_x)

    return selected

def _shrink_maximal(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            x = selected[idx]
            candidate = selected[:idx] + selected[idx+1:]
            
            if _is_maximal_sidon(candidate, N):
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
    
    # Increase restarts since greedy is faster
    num_restarts = 5000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Generate initial maximal set using blocking heuristic
        sel = _greedy_blocking_maximal_fast(rng, N)
        
        if not sel:
            continue
            
        # Shrink it
        sel = _shrink_maximal(sel, N, deadline)
        
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            if best_size <= 16:
                break

    if best is None:
        # Fallback
        sel = _greedy_blocking_maximal_fast(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
