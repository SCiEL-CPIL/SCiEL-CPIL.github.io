"""Construct a 4-AP-free subset of [1,512] using Cantor construction + local search."""
import time
import random

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _cantor_set():
    """Select numbers in [1,512] whose base-3 representation contains no digit '2'.
    This gives a 4-AP-free set (generalized Cantor/Salem-Spencer construction)."""
    s = set()
    for x in range(1, N + 1):
        ok = True
        v = x
        while v > 0:
            if v % 3 == 2:
                ok = False
                break
            v //= 3
        if ok:
            s.add(x)
    return s

def _has_4ap(s, x):
    """Check if adding x to set s creates a nonconstant 4-AP.
    A 4-AP is a, a+d, a+2d, a+3d with d>=1.
    x could be the 1st, 2nd, 3rd, or 4th term."""
    # x is 1st term: x, x+d, x+2d, x+3d all in s (x not yet in s)
    # We need x+d, x+2d, x+3d in s
    for d in range(1, (N - x) // 3 + 1):
        if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
            return True
    # x is 2nd term: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in s
    for d in range(1, x // 1 + 1):
        if d >= x:
            break
        if (x - d) in s and (x + d) in s and (x + 2*d) in s:
            return True
    # x is 3rd term: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in s
    for d in range(1, x // 2 + 1):
        if (x - 2*d) in s and (x - d) in s and (x + d) in s:
            return True
    # x is 4th term: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in s
    for d in range(1, x // 3 + 1):
        if (x - 3*d) in s and (x - 2*d) in s and (x - d) in s:
            return True
    return False

def _local_search(s, deadline):
    """Try to add elements to s via local search (add if no 4-AP created)."""
    s = set(s)
    candidates = [x for x in range(1, N + 1) if x not in s]
    random.shuffle(candidates)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if x in s:
                continue
            if not _has_4ap(s, x):
                s.add(x)
                improved = True
    return s

def _try_remove_and_add(s, deadline):
    """Try removing one element to enable adding two or more others."""
    s = set(s)
    current_size = len(s)
    candidates = [x for x in range(1, N + 1) if x not in s]
    random.shuffle(candidates)
    
    # For each element in s, try removing it and then greedily adding
    for elem in list(s):
        if time.time() > deadline:
            break
        s.remove(elem)
        added = 0
        for x in candidates:
            if x in s:
                continue
            if not _has_4ap(s, x):
                s.add(x)
                added += 1
        if len(s) > current_size:
            current_size = len(s)
        else:
            # Revert: put elem back, remove what was added
            s.discard(elem)
            # This is complex; simpler to just track best
            s = set(s)  # keep as is if better, else revert
            # Actually let's just track the best set
            if len(s) < current_size:
                # revert
                pass
        # Continue searching
    return s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Cantor set
    s = _cantor_set()
    best = set(s)
    best_size = len(s)
    
    # Greedy add
    s = _local_search(s, deadline)
    if len(s) > best_size:
        best = set(s)
        best_size = len(s)
    
    # Try multiple random restarts with different initial perturbations
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # Start from best, try random local search
        s2 = set(best)
        # Randomly remove a few elements to explore
        to_remove = rng.sample(sorted(s2), min(3, len(s2) // 10))
        for r in to_remove:
            s2.discard(r)
        s2 = _local_search(s2, deadline)
        if len(s2) > best_size:
            best = set(s2)
            best_size = len(s2)
        
        # Also try fresh random starts
        if attempts % 5 == 0:
            s3 = _cantor_set()
            # Randomly swap some elements
            in_set = sorted(s3)
            out_set = [x for x in range(1, N+1) if x not in s3]
            for _ in range(rng.randint(1, 5)):
                if in_set and out_set:
                    rem = rng.choice(in_set)
                    s3.discard(rem)
                    in_set.remove(rem)
                    add = rng.choice(out_set)
                    s3.add(add)
                    out_set.remove(add)
            s3 = _local_search(s3, deadline)
            if len(s3) > best_size:
                best = set(s3)
                best_size = len(s3)
    
    return sorted(best)
