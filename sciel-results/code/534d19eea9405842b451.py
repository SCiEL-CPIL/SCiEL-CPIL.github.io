import math
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = 130.0  # Safety margin under the 150s wall-clock kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, _ in classes)

def _find_residue(classes, m):
    """
    Find a residue r for modulus m that is disjoint from all existing classes.
    Returns r if found, else None.
    """
    # Pre-compute gcds and forbidden residues for speed?
    # For m up to 200, simple loop is fine.
    for r in range(m):
        ok = True
        for (mi, ri) in classes:
            g = gcd(m, mi)
            if (r % g) == (ri % g):
                ok = False
                break
        if ok:
            return r
    return None

def _add_with_repair(classes, m, rng, max_retries=50):
    """
    Try to add modulus m to classes.
    If direct addition fails, try reassigning residues of existing moduli.
    Returns True if m was added, False otherwise.
    """
    # 1. Try direct addition
    r = _find_residue(classes, m)
    if r is not None:
        classes.append([m, r])
        return True
    
    # 2. Repair phase
    # We need to change some existing residues to make room for m.
    # Identify which existing moduli are "blocking" m.
    # A modulus mi blocks m if the set of residues forbidden by mi
    # covers a large part of [0, m).
    
    n = len(classes)
    if n == 0:
        return False
        
    # Randomly select a few existing moduli to perturb
    # We prioritize moduli that share a factor with m
    candidates = [i for i in range(n) if gcd(classes[i][0], m) > 1]
    if not candidates:
        candidates = list(range(n))
    
    for _ in range(max_retries):
        if _time_left() < 5:
            return False
            
        # Pick a random candidate to change
        idx = rng.choice(candidates)
        mi, ri_old = classes[idx]
        
        # Try a new random residue for mi
        ri_new = rng.randrange(mi)
        if ri_new == ri_old:
            continue
            
        # Check if ri_new is valid for mi with all OTHER classes
        valid_for_i = True
        for j in range(n):
            if j == idx:
                continue
            mj, rj = classes[j]
            if not _pair_ok(mi, ri_new, mj, rj):
                valid_for_i = False
                break
        
        if not valid_for_i:
            continue
            
        # Tentatively apply change
        classes[idx][1] = ri_new
        
        # Now try to add m again
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            return True
            
        # Revert change
        classes[idx][1] = ri_old
        
    return False

def _local_search(classes, rng, budget):
    """
    Iteratively try to add missing moduli in increasing order.
    """
    t0 = time.time()
    used = set(m for m, _ in classes)
    
    # We want to add moduli from 20 to 200 that are not yet used
    # Prioritize small moduli
    missing = [m for m in range(20, 201) if m not in used]
    
    # Sort missing by value (ascending) to prioritize high score contributions
    # Actually, we should try to add them in order of 1/m (descending), i.e., m ascending
    for m in missing:
        if _time_left() < 5 or (time.time() - t0) > budget:
            break
            
        # Try to add m
        if _add_with_repair(classes, m, rng, max_retries=100):
            used.add(m)
            # After adding, we might have changed residues, which is fine.
            # We continue to the next missing modulus.
            
    return classes

def construct():
    rng = random.Random(42)
    best = None
    best_score = -1.0
    
    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]
                
    # Strategy 1: Greedy from 20 to 200 with repair
    c = []
    c = _local_search(c, rng, min(40.0, _time_left() - 20))
    consider(c)
    
    # Strategy 2: Random Restart Greedy
    # Start with a random subset or random order?
    # Better: Start with small moduli, but randomize the residue selection order in repair.
    
    while _time_left() > 15:
        c = []
        # Greedy add in increasing order, but with random repair
        c = _local_search(c, rng, min(10.0, max(2.0, _time_left() - 15)))
        consider(c)
        
        # Occasionally, try a "dense" start: pick all even moduli first?
        # Or pick moduli with many small prime factors?
        
    # Fallback
    if best is None:
        best = [[20, 0]]
        
    return best
