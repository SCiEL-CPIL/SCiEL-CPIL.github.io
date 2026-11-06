import time
import math
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _is_disjoint(m1, r1, m2, r2):
    """Check if residue classes (m1, r1) and (m2, r2) are disjoint."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _score(classes):
    """Compute the sum of 1/m for the given classes."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _can_add(classes, m_new, r_new):
    """Check if a new residue class can be added to the existing set."""
    for m, r in classes:
        if not _is_disjoint(m, r, m_new, r_new):
            return False
    return True

def _greedy_construct():
    """
    Greedy construction: try to add residue classes in order of increasing modulus.
    For each modulus, try all residues and pick the first one that is disjoint
    from all existing classes.
    """
    classes = []
    for m in range(20, 201):
        added = False
        for r in range(m):
            if _can_add(classes, m, r):
                classes.append([m, r])
                added = True
                break
    return classes

def _local_search(classes, time_budget=10.0):
    """
    Local search: try to improve the score by swapping residues or adding/removing classes.
    We try to replace some moduli with smaller ones (higher 1/m) if possible.
    """
    start_time = time.time()
    classes = [list(c) for c in classes]
    best_score = _score(classes)
    
    # Try to improve by trying to add missing small moduli
    # and by swapping residues to free up constraints for smaller moduli
    for m_target in range(20, 201):
        if time.time() - start_time > time_budget:
            break
        if any(c[0] == m_target for c in classes):
            continue
        # Try to add m_target with some residue
        for r_target in range(m_target):
            if _can_add(classes, m_target, r_target):
                classes.append([m_target, r_target])
                new_score = _score(classes)
                if new_score > best_score:
                    best_score = new_score
                break
    
    # Try to swap residues to allow adding more small moduli
    # This is a simple local search
    for _ in range(100):  # limited iterations
        if time.time() - start_time > time_budget:
            break
        improved = False
        for i, (m1, r1) in enumerate(classes):
            for r_new in range(m1):
                if r_new == r1:
                    continue
                # Try changing residue of class i
                old_r = classes[i][1]
                classes[i][1] = r_new
                # Check if still disjoint
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if i == j:
                        continue
                    if not _is_disjoint(m1, r_new, m2, r2):
                        valid = False
                        break
                if valid:
                    # Now try to add a small modulus that was previously blocked
                    for m_add in range(20, 201):
                        if any(c[0] == m_add for c in classes):
                            continue
                        for r_add in range(m_add):
                            if _can_add(classes, m_add, r_add):
                                classes.append([m_add, r_add])
                                new_score = _score(classes)
                                if new_score > best_score:
                                    best_score = new_score
                                else:
                                    # Revert if no improvement
                                    classes.pop()
                                classes[i][1] = r_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                else:
                    classes[i][1] = old_r
        if not improved:
            break
    
    return classes

def _simulated_annealing(classes, time_budget=30.0):
    """
    Simulated annealing to improve the score.
    Moves: change a residue, add a class, remove a class.
    """
    start_time = time.time()
    classes = [list(c) for c in classes]
    best_classes = [list(c) for c in classes]
    best_score = _score(classes)
    current_score = best_score
    
    T = 0.1
    T_min = 0.001
    cooling = 0.95
    
    while time.time() - start_time < time_budget and T > T_min:
        # Random move
        move_type = random.random()
        
        if move_type < 0.6:
            # Change a residue
            if classes:
                i = random.randrange(len(classes))
                m, r = classes[i]
                old_r = r
                new_r = random.randrange(m)
                if new_r == old_r:
                    continue
                classes[i][1] = new_r
                # Check validity
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if i == j:
                        continue
                    if not _is_disjoint(m, new_r, m2, r2):
                        valid = False
                        break
                if not valid:
                    classes[i][1] = old_r
                    continue
                new_score = _score(classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [list(c) for c in classes]
                else:
                    classes[i][1] = old_r
                    
        elif move_type < 0.8:
            # Add a class
            m_add = random.randrange(20, 201)
            if any(c[0] == m_add for c in classes):
                continue
            r_add = random.randrange(m_add)
            if _can_add(classes, m_add, r_add):
                classes.append([m_add, r_add])
                new_score = _score(classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [list(c) for c in classes]
                else:
                    classes.pop()
                    
        else:
            # Remove a class
            if len(classes) > 1:
                i = random.randrange(len(classes))
                removed = classes.pop(i)
                new_score = _score(classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [list(c) for c in classes]
                else:
                    classes.insert(i, removed)
        
        T *= cooling
    
    return best_classes

def construct():
    """
    Main construction function.
    Uses greedy initialization followed by local search and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Phase 1: Greedy construction
    classes = _greedy_construct()
    score = _score(classes)
    
    if time.time() > deadline - 30:
        return classes
    
    # Phase 2: Local search
    classes = _local_search(classes, time_budget=15.0)
    score = _score(classes)
    
    if time.time() > deadline - 10:
        return classes
    
    # Phase 3: Simulated annealing
    classes = _simulated_annealing(classes, time_budget=60.0)
    score = _score(classes)
    
    return classes

# EVOLVE-BLOCK-END
