"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _divisors_of(base):
    """Return all divisors of base greater than 1, sorted ascending."""
    divs = []
    i = 1
    while i * i <= base:
        if base % i == 0:
            if i > 1:
                divs.append(i)
            j = base // i
            if j != i and j > 1:
                divs.append(j)
        i += 1
    return sorted(divs)

def _bitmask_for(m, r, base):
    """
    Precompute a bitmask (as a Python int) where bit i is 1 iff i % m == r.
    Uses the fact that base=51975 and m divides base, so the pattern is periodic.
    """
    mask = 0
    for pos in range(r, base, m):
        mask |= (1 << pos)
    return mask

def _count_uncovered(uncovered_mask):
    """Count bits set in uncovered_mask."""
    return bin(uncovered_mask).count('1')

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    base = 51975
    divisors = _divisors_of(base)
    
    # Precompute bitmasks for each divisor and each residue
    # masks[m][r] = bitmask of positions where pos % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            masks[m][r] = _bitmask_for(m, r, base)
    
    # Initial uncovered: all residues
    full_mask = (1 << base) - 1
    uncovered = full_mask
    
    # Greedy construction: at each step, pick (m, r) that covers the most uncovered residues
    # We process moduli in increasing order for determinism, but allow multiple classes per modulus?
    # No: distinct moduli. So each modulus can be used at most once.
    
    # Actually, re-reading the task: "classes [[m, r], ...] with distinct moduli m > 1"
    # So each modulus appears at most once in the list.
    
    # Greedy: for each modulus (in some order), pick the residue that covers most uncovered.
    # But we can choose which moduli to include and in what order.
    
    # Strategy: 
    # 1. Greedy build: repeatedly pick the (m, r) pair that maximizes newly covered residues,
    #    where m is not yet used.
    # 2. Local search: try swapping residues for used moduli, or adding/removing moduli.
    
    def greedy_build():
        used_moduli = set()
        classes = []
        unc = full_mask
        for _ in range(len(divisors)):
            if time.time() - start_time > DEADLINE:
                break
            best_gain = 0
            best_m = None
            best_r = None
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    mask = masks[m][r]
                    gain = _count_uncovered(unc & mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            if best_m is None or best_gain == 0:
                break
            used_moduli.add(best_m)
            classes.append([best_m, best_r])
            unc &= ~masks[best_m][best_r]
        return classes, unc
    
    classes, uncovered = greedy_build()
    
    if time.time() - start_time > DEADLINE:
        return classes
    
    # Local search phase: try to improve by changing residues of existing moduli
    # or by swapping out a modulus for a different one
    
    def current_uncovered(classes):
        unc = full_mask
        for m, r in classes:
            unc &= ~masks[m][r]
        return unc
    
    best_classes = classes
    best_uncovered = uncovered
    best_score = _count_uncovered(uncovered)
    
    # Local search: iterate, trying single moves
    used_moduli = set(m for m, r in classes)
    
    for iteration in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        
        # Try changing the residue of each used modulus
        for i, (m, r) in enumerate(classes):
            for new_r in range(m):
                if new_r == r:
                    continue
                # Compute new uncovered
                new_unc = full_mask
                for j, (m2, r2) in enumerate(classes):
                    if j == i:
                        new_unc &= ~masks[m2][new_r]
                    else:
                        new_unc &= ~masks[m2][r2]
                score = _count_uncovered(new_unc)
                if score < best_score:
                    best_score = score
                    best_classes = classes[:]
                    best_classes[i] = [m, new_r]
                    classes = best_classes[:]
                    uncovered = new_unc
                    improved = True
                    break
        
        # Try adding a new modulus
        if not improved:
            best_add_gain = 0
            best_add_m = None
            best_add_r = None
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    gain = _count_uncovered(uncovered & masks[m][r])
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_m = m
                        best_add_r = r
            if best_add_m is not None and best_add_gain > 0:
                classes.append([best_add_m, best_add_r])
                used_moduli.add(best_add_m)
                uncovered &= ~masks[best_add_m][best_add_r]
                new_score = _count_uncovered(uncovered)
                if new_score < best_score:
                    best_score = new_score
                    best_classes = classes[:]
                improved = True
        
        # Try removing a modulus (if it doesn't hurt)
        if not improved:
            for i in range(len(classes)):
                m, r = classes[i]
                # Temporarily remove
                new_unc = full_mask
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        new_unc &= ~masks[m2][r2]
                score = _count_uncovered(new_unc)
                if score <= best_score:
                    # Removing doesn't make it worse (or improves)
                    # Only remove if it actually improves or we want to simplify
                    if score < best_score:
                        classes.pop(i)
                        used_moduli.discard(m)
                        uncovered = new_unc
                        best_score = score
                        best_classes = classes[:]
                        improved = True
                        break
        
        if not improved:
            break
    
    # Final check
    if time.time() - start_time > DEADLINE:
        return best_classes
    
    # Verify best_classes
    final_unc = full_mask
    for m, r in best_classes:
        final_unc &= ~masks[m][r]
    final_score = _count_uncovered(final_unc)
    
    if final_score <= best_score:
        return best_classes
    else:
        return classes

# EVOLVE-BLOCK-END
