import time

def _check_valid(S):
    """Check if set S has no forbidden triple."""
    n = len(S)
    for a in S:
        for b in S:
            if b <= a:
                continue
            ab = a + b
            if ab not in S:
                continue
            for c in S:
                if c <= b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
    return True

def _greedy_add(cur, deadline):
    """Greedily add elements to cur until no more can be added or time runs out."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 129):
            if x in cur:
                continue
            if (x + 1) in cur or (x + 2) in cur:
                pass  # still might be valid, just check
            cand = cur | {x}
            if _check_valid(cand):
                cur = cand
                changed = True
                break
    return cur

def _local_search(cur, deadline):
    """Try removing one element and re-adding others to escape local optima."""
    best = cur
    best_size = len(cur)
    while time.time() < deadline:
        improved = False
        for remove_x in sorted(best):
            if time.time() > deadline:
                break
            reduced = best - {remove_x}
            # Try to add back other elements
            cand = _greedy_add(set(reduced), deadline)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
                improved = True
                break
        if not improved:
            break
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Use a structured initial seed: sum-free blocks
    # Numbers > 64 are sum-free among themselves (sum > 128)
    # Mix in some lower numbers that don't create conflicts
    base = set(range(65, 129))  # 64 numbers, all pair sums > 128 so safe
    
    # Try to add some numbers from 1..64
    # A number x in 1..64 is safe to add if for any two existing elements a,b
    # we don't create a forbidden configuration
    cur = set(base)
    
    # Greedily add from 1..64
    for x in range(1, 65):
        if time.time() > deadline:
            break
        cand = cur | {x}
        if _check_valid(cand):
            cur = cand
    
    # Now do a full greedy pass over all 1..128
    cur = _greedy_add(cur, deadline)
    
    # Local search polish
    cur = _local_search(cur, deadline)
    
    return [int(v) for v in sorted(cur)]
