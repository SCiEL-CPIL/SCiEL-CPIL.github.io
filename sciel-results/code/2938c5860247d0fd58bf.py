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

def _greedy_blocking_maximal(rng, N):
    """
    Generate a maximal Sidon set by greedily adding the element
    that blocks the most other unblocked elements.
    """
    selected = []
    sel_set = set()
    sums = set()
    
    # Precompute potential blocking impact?
    # We just simulate.
    
    while True:
        # Find unblocked elements
        unblocked = []
        blocked = set()
        
        # Calculate blocked set from current sums and selected
        for s in sums:
            if s % 2 == 0:
                x_candidate = s // 2
                if 1 <= x_candidate <= N:
                    blocked.add(x_candidate)
            for a in selected:
                x_candidate = s - a
                if 1 <= x_candidate <= N:
                    blocked.add(x_candidate)
        
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if x in blocked:
                continue
            unblocked.append(x)
            
        if not unblocked:
            break
            
        # Choose x that maximizes blocking
        best_x = None
        best_score = -1
        best_new_blocked = set()
        
        # To speed up, we can sample if unblocked is large, but N=256 is small
        for x in unblocked:
            # Simulate adding x
            # New sums: 2x, x+a for a in selected
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
                
            # Check validity (Sidon)
            # Since we only add if x was unblocked, is it guaranteed Sidon?
            # Unblocked means 2x not in sums and x+a not in sums.
            # So new sums are distinct from old sums.
            # Are new sums distinct from each other?
            # 2x vs x+a: 2x = x+a => x=a. If x in selected, x is blocked (2x in sums? No, if x in selected, x is in sel_set, so not in unblocked).
            # So x not in selected.
            # x+a1 = x+a2 => a1=a2.
            # So new sums are distinct.
            # So validity is guaranteed.
            
            # Calculate how many NEW elements are blocked
            # New blocked set
            new_blocked = set()
            for s in new_sums:
                if s % 2 == 0:
                    xc = s // 2
                    if 1 <= xc <= N:
                        new_blocked.add(xc)
                for a in selected + [x]: # Wait, sums include x+a.
                     # If s = x+a, then s-a = x. x is already in sel_set (conceptually).
                     # We care about blocking OTHER elements.
                     pass
            
            # Simpler:
            # An element y is blocked if 2y in new_sums or y+a in new_sums for a in selected+{x}.
            # We want to count y in unblocked \ {x} that become blocked.
            
            count = 0
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is blocked by new sums
                is_blocked = False
                if 2 * y in new_sums:
                    is_blocked = True
                else:
                    for a in selected:
                        if y + a in new_sums:
                            is_blocked = True
                            break
                    if not is_blocked:
                        if y + x in new_sums:
                            is_blocked = True
                            
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
        for a in selected[:-1]: # selected[:-1] is old selected
            sums.add(best_x + a)
            
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
    
    num_restarts = 1000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Generate initial maximal set using blocking heuristic
        sel = _greedy_blocking_maximal(rng, N)
        
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
        sel = _greedy_blocking_maximal(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
