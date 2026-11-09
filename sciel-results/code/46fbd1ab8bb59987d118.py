"""Extended Sidon set search with full validation and iterative local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_sidon(a):
    """Check if a sorted list is a valid Sidon set. Returns True if valid."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s = a[i] + a[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_construct(prescribed, start_x=12, end_x=284):
    """Greedily construct a Sidon set starting with prescribed elements."""
    a = sorted(prescribed)
    sums = set()
    # Pre-calculate sums for initial set
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    
    for x in range(start_x, end_x + 1):
        if x in a:
            continue
        ok = True
        for y in a:
            if (x + y) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            a.append(x)
            # Update sums
            for y in a:
                if y != x:
                    sums.add(x + y)
            sums.add(2 * x)
            a.sort() # Keep sorted for consistency, though greedy scans in order
    return a

def _random_perturb(a, prescribed, num_removals=2):
    """Remove random non-prescribed elements to create a new starting point."""
    a = list(a)
    removable = [x for x in a if x not in prescribed]
    if not removable:
        return a
    k = min(num_removals, len(removable))
    to_remove = random.sample(removable, k)
    for x in to_remove:
        a.remove(x)
    return a

def _try_extend(a, prescribed, deadline):
    """Attempt to extend the set greedily from its current state.
    Returns the extended set if valid, else None."""
    # We try to add elements in increasing order.
    # To be more robust, we can try multiple random orders or just standard greedy.
    # Let's stick to standard greedy for speed, but we validate at the end.
    current = sorted(a)
    sums = set()
    n = len(current)
    for i in range(n):
        for j in range(i, n):
            sums.add(current[i] + current[j])
            
    for x in range(1, 285):
        if time.time() > deadline:
            break
        if x in current:
            continue
        ok = True
        for y in current:
            if (x + y) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            current.append(x)
            for y in current:
                if y != x:
                    sums.add(x + y)
            sums.add(2 * x)
            current.sort()
    return current

def construct():
    deadline = time.time() + 145  # Safety margin
    prescribed = {1, 2, 5, 11}
    
    # Initial greedy construction
    best = _greedy_construct(list(prescribed))
    best_n = len(best)
    
    # Validate best
    if not _is_sidon(best):
        # Fallback to just prescribed if greedy failed (shouldn't happen)
        best = sorted(prescribed)
        best_n = 4

    random.seed(42)
    
    # Iterative Local Search
    current = list(best)
    
    while time.time() < deadline:
        # Perturb
        # Strategy: Remove 1-3 random non-prescribed elements
        num_removals = random.randint(1, 3)
        perturbed = _random_perturb(current, prescribed, num_removals)
        
        # Extend
        candidate = _try_extend(perturbed, prescribed, deadline)
        
        # Validate
        if candidate and _is_sidon(candidate):
            if len(candidate) > best_n:
                best = candidate
                best_n = len(best)
                current = list(best) # Update current to best for future perturbations
            else:
                # Even if not better, we might use it as a new current if it's valid
                # to explore different local minima, but sticking to best is safer for size
                pass
        else:
            # If extension failed or produced invalid set, stick to current
            pass
            
    return best

# EVOLVE-BLOCK-END
