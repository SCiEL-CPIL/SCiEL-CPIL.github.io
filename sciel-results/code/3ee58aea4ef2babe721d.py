import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, selected, gcd_table):
    """Check if new (r_new, m_new) is disjoint from all selected classes."""
    for (r, m) in selected:
        g = gcd_table.get((m_new, m), 1)
        if g == 1:
            return False
        if (r_new - r) % g == 0:
            return False
    return True

def _local_search(selected, gcd_table, deadline):
    """
    Try to improve the solution by:
    1. Removing a class and re-adding it with a different residue
    2. Trying to add a currently-excluded modulus with a new residue
    3. Swapping residues of two classes
    Returns the improved list.
    """
    current = [tuple(x) for x in selected]
    current = sorted(current, key=lambda x: x[1])
    
    # Precompute which moduli are already selected
    selected_moduli = set(m for r, m in current)
    
    # All moduli 2..128
    all_moduli = list(range(2, 129))
    excluded_moduli = [m for m in all_moduli if m not in selected_moduli]
    
    improved = False
    max_iter = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Strategy 1: Try to add an excluded modulus
        for m_new in random.sample(excluded_moduli, min(len(excluded_moduli), 50)):
            # Check if m_new is coprime to any selected modulus
            skip = False
            for (r, m) in current:
                if _gcd(m_new, m) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            # Try to find a residue
            found = False
            for r_new in random.sample(range(m_new), m_new):
                if _can_add(r_new, m_new, current, gcd_table):
                    current.append((r_new, m_new))
                    selected_moduli.add(m_new)
                    excluded_moduli.remove(m_new)
                    improved = True
                    found = True
                    break
            if found:
                break
        
        # Strategy 2: Try to re-resolve a selected modulus
        if not improved:
            idx = random.randint(0, len(current) - 1)
            r_old, m_old = current[idx]
            current.pop(idx)
            
            found = False
            for r_new in random.sample(range(m_old), m_old):
                if r_new == r_old:
                    continue
                if _can_add(r_new, m_old, current, gcd_table):
                    current.append((r_new, m_old))
                    found = True
                    improved = True
                    break
            
            if not found:
                current.append((r_old, m_old))
        
        # Strategy 3: Try swapping residues
        if not improved and len(current) >= 2:
            i, j = random.sample(range(len(current)), 2)
            r1, m1 = current[i]
            r2, m2 = current[j]
            current[i] = (r2, m1)
            current[j] = (r1, m2)
            
            valid = True
            for a in range(len(current)):
                for b in range(a + 1, len(current)):
                    ra, ma = current[a]
                    rb, mb = current[b]
                    g = gcd_table.get((ma, mb), 1)
                    if g == 1 or (ra - rb) % g == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                improved = True
            else:
                current[i] = (r1, m1)
                current[j] = (r2, m2)
    
    return current

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin for 90s limit
    
    # All moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    best_result = None
    best_count = 0
    
    # Strategy: Greedy with multiple random orderings + local search
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        random.seed(attempt * 7 + 13)
        perm = all_moduli[:]
        random.shuffle(perm)
        
        selected = []
        
        for m in perm:
            if time.time() > deadline:
                break
            
            # Check if m is coprime to any already selected modulus
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            # Try to find a residue for m that works
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if _can_add(r, m, selected, gcd_table):
                    selected.append((r, m))
                    found = True
                    break
            
            # If not found, skip this modulus (greedy)
        
        # Apply local search to improve
        if len(selected) > 0:
            improved = _local_search(selected, gcd_table, deadline)
            if len(improved) > len(selected):
                selected = improved
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = selected[:]
        
        # Early exit if we find a very good solution
        if best_count >= 50:
            break
    
    # If greedy didn't find anything good, try a more structured approach
    # Focus on powers of 2 or other structured sets
    
    if best_result is None or best_count < 10:
        # Structured fallback: Use moduli that are powers of 2
        structured = []
        for k in range(7):
            m = 1 << (k + 1)
            if m <= 128:
                r = (1 << k) - 1  # e.g., 1, 3, 7, 15, 31, 63, 127
                structured.append((r, m))
        
        # Verify
        valid = True
        for i in range(len(structured)):
            for j in range(i + 1, len(structured)):
                if not _disjoint(structured[i][0], structured[i][1], structured[j][0], structured[j][1]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(structured) > best_count:
            best_result = structured
            best_count = len(structured)
    
    # Final fallback
    if not best_result:
        best_result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return [[r, m] for r, m in best_result]
