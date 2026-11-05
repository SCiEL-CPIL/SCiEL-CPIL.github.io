import random
import numpy as np

N = 256

def _is_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _is_maximal(s, n):
    """Check if Sidon set s is inclusion-maximal in [1, n]."""
    s_set = set(s)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
    for x in range(1, n + 1):
        if x in s_set:
            continue
        # Check if adding x would break Sidon property
        # We need to check if x+a conflicts with existing sums or other new sums
        # But for maximality, we just need to know if x CAN be added.
        # x can be added if:
        # 1. x+x not in sums
        # 2. x+a not in sums for all a in s
        # 3. x+a1 != x+a2 for a1!=a2 (always true if a1!=a2)
        # 4. x+a1 != x+x (always true if a1 != x, which is true since x not in s)
        
        # So we just need to check:
        # 2x not in sums
        # (x + a) not in sums for all a in s
        
        if 2 * x in sums:
            continue
        can_add = True
        for a in s:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            return False
    return True

def _greedy_maximal(order):
    """Build a maximal Sidon set by processing elements in given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    # Now make it maximal: try adding remaining elements
    s_set = set(selected)
    for x in order:
        if x in s_set:
            continue
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
            s_set.add(x)
    return selected

def _minimize_maximal(s, n):
    """Try to remove elements from a maximal Sidon set while keeping it maximal."""
    s = list(s)
    changed = True
    while changed:
        changed = False
        # Try removing each element
        # To be efficient, we can just check one by one
        for i in range(len(s)):
            # Temporarily remove s[i]
            candidate = s[:i] + s[i+1:]
            if _is_maximal(candidate, n):
                s = candidate
                changed = True
                break # Restart the loop as indices changed
    return s

def construct():
    best_set = None
    best_size = N + 1
    
    # 1. Get an initial greedy solution
    greedy = _greedy_maximal(list(range(1, N + 1)))
    greedy_min = _minimize_maximal(greedy, N)
    if _is_sidon(greedy_min) and _is_maximal(greedy_min, N):
        best_set = greedy_min
        best_size = len(greedy_min)
    
    # 2. Random Restart Greedy with Minimization
    # We run many restarts. Each restart:
    # a) Generate random maximal Sidon set
    # b) Minimize it (remove elements if possible)
    # c) Update best if smaller
    
    # Time budget management: assume ~0.001s per restart+minimize
    # 30s limit -> ~20000 restarts possible.
    # Let's do 2000 restarts to be safe, and increase if time permits?
    # No, just do a fixed number that is fast.
    
    for restart in range(2000):
        order = list(range(1, N + 1))
        random.shuffle(order)
        current = _greedy_maximal(order)
        
        # Minimize
        current_min = _minimize_maximal(current, N)
        
        if len(current_min) < best_size:
            best_size = len(current_min)
            best_set = current_min
            
            # Early exit if we hit a theoretical lower bound?
            # Lower bound for maximal Sidon set size in [1, N] is ~ sqrt(N)?
            # For N=256, sqrt=16. If we find 16, it's likely optimal or close.
            if best_size <= 16:
                break

    if best_set is None or not _is_sidon(best_set) or not _is_maximal(best_set, N):
        # Fallback
        return greedy
    
    return best_set
