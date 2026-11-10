"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered pair sums distinct)."""
    n = len(selected)
    sums = set()
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_sidon_fast(selected, N):
    """Check if selected is Sidon AND maximal (no missing element can be added)."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False  # Not Sidon
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
    Optimized with precomputed difference set.
    """
    selected = []
    sel_set = set()
    sums = set()
    diff_set = set()  # All differences b - a for a, b in selected
    
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
            count = 0
            sx = 2 * x
            
            for y in unblocked:
                if y == x:
                    continue
                is_blocked = False
                
                # Condition 1: 2y == x + a for some a in selected
                two_y = 2 * y
                for a in selected:
                    if x + a == two_y:
                        is_blocked = True
                        break
                if is_blocked:
                    count += 1
                    continue
                    
                # Condition 2: y + a == 2x for some a in selected
                for a in selected:
                    if y + a == sx:
                        is_blocked = True
                        break
                if is_blocked:
                    count += 1
                    continue
                    
                # Condition 3: y - x is in diff_set (i.e., y = x + b - a for some a,b in selected)
                if (y - x) in diff_set:
                    is_blocked = True
                
                if is_blocked:
                    count += 1
                    
            if count > best_score:
                best_score = count
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        for a in selected:
            diff_set.add(best_x - a)
            diff_set.add(a - best_x)
        selected.append(best_x)
        sel_set.add(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
            
    return selected

def _shrink_maximal_fast(selected, N, deadline):
    """Try to remove elements while maintaining Sidon AND maximality."""
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

def _local_search_swap(selected, N, deadline, rng):
    """Try to replace elements to reduce size while maintaining Sidon+maximal."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        sel_set = set(selected)
        for i in range(n):
            if time.time() > deadline:
                return selected
            a = selected[i]
            # Try removing a and adding some other element
            base = selected[:i] + selected[i+1:]
            base_set = set(base)
            # Try all candidates not in base
            candidates = [x for x in range(1, N+1) if x not in base_set]
            rng.shuffle(candidates)
            for c in candidates[:50]:  # Limit candidates to save time
                candidate = base + [c]
                if _is_maximal_sidon_fast(candidate, N):
                    # Check if this actually helps (same size, but maybe enables further removal)
                    # For now, just accept if valid
                    selected = candidate
                    improved = True
                    break
            if improved:
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
