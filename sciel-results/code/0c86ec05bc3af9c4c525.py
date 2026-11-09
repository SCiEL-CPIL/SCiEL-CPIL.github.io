import time
import math
import random
from fractions import Fraction

# Precompute gcds for moduli 20 to 200
MOD_MIN = 20
MOD_MAX = 200
MODS = list(range(MOD_MIN, MOD_MAX + 1))
N_MODS = len(MODS)
MOD_INDEX = {m: i for i, m in enumerate(MODS)}

# gcds[i][j] = gcd(MODS[i], MODS[j])
gcds = [[0] * N_MODS for _ in range(N_MODS)]
for i in range(N_MODS):
    for j in range(i, N_MODS):
        g = math.gcd(MODS[i], MODS[j])
        gcds[i][j] = g
        gcds[j][i] = g

def score(classes):
    """Calculate the sum of 1/m for the given classes."""
    return sum(1.0 / m for m, r in classes)

def is_compatible(m1, r1, m2, r2):
    """Check if (m1, r1) and (m2, r2) are disjoint."""
    i1 = MOD_INDEX[m1]
    i2 = MOD_INDEX[m2]
    g = gcds[i1][i2]
    return (r1 % g) != (r2 % g)

def is_valid(classes):
    """Check if all pairs in classes are disjoint."""
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not is_compatible(m1, r1, m2, r2):
                return False
    return True

def greedy_restart(seed):
    """
    Perform a greedy construction with randomized residue selection.
    Moduli are processed in ascending order.
    """
    random.seed(seed)
    selected = []
    
    # Process moduli in ascending order to prioritize small m
    for m in MODS:
        # Generate random order of residues
        residues = list(range(m))
        random.shuffle(residues)
        
        # Find first valid residue
        for r in residues:
            valid = True
            for m2, r2 in selected:
                if not is_compatible(m, r, m2, r2):
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                break
                
    return selected

def local_search(classes, seed):
    """
    Attempt to improve the solution by swapping one modulus for another.
    We try to replace a larger modulus with a smaller one if possible.
    """
    if not classes:
        return classes
    
    random.seed(seed)
    current_classes = [list(c) for c in classes]
    current_score = score(current_classes)
    
    # Create a set of selected moduli for quick lookup
    selected_mods = {m for m, r in current_classes}
    unselected_mods = [m for m in MODS if m not in selected_mods]
    
    improved = True
    while improved:
        improved = False
        # Shuffle unselected and selected for diversity
        random.shuffle(unselected_mods)
        random.shuffle(current_classes)
        
        for m_new in unselected_mods:
            if m_new in selected_mods:
                continue
            # Only consider replacing with smaller modulus (higher weight)
            # Actually, just check if replacing improves score
            for idx, (m_old, r_old) in enumerate(current_classes):
                if m_new >= m_old:
                    continue # No improvement possible in score if m_new >= m_old
                
                # Temporarily remove m_old
                remaining = [c for i, c in enumerate(current_classes) if i != idx]
                
                # Try to find a valid residue for m_new
                residues = list(range(m_new))
                random.shuffle(residues)
                
                found = False
                for r_new in residues:
                    valid = True
                    for m2, r2 in remaining:
                        if not is_compatible(m_new, r_new, m2, r2):
                            valid = False
                            break
                    if valid:
                        # Accept the swap
                        current_classes[idx] = [m_new, r_new]
                        selected_mods.remove(m_old)
                        selected_mods.add(m_new)
                        unselected_mods.append(m_old)
                        if m_new in unselected_mods:
                            unselected_mods.remove(m_new)
                        
                        new_score = score(current_classes)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                        found = True
                        break
                
                if found:
                    break # Move to next m_new? Or restart loop?
                    # Breaking inner loop, continuing outer
            if improved:
                break # Restart the while loop to re-evaluate with new state

    return [tuple(c) for c in current_classes]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_classes = []
    best_score = 0.0
    
    # Initial deterministic greedy (smallest residues)
    # This is equivalent to seed=0 with a specific shuffle, but let's just run random restarts
    
    restart_count = 0
    while time.time() < deadline:
        seed = restart_count
        candidate = greedy_restart(seed)
        
        # Local search on the candidate
        # Only do local search if we have time
        if time.time() < deadline - 10.0:
            candidate = local_search(candidate, seed)
            
        s = score(candidate)
        
        if s > best_score:
            best_score = s
            best_classes = candidate
            # print(f"New best: {best_score} with {len(best_classes)} classes") # Uncomment for debug
            
        restart_count += 1
        
        # If we've done a lot of restarts, maybe slow down or stop?
        # The loop condition handles time.
        
    # Final validation
    if not is_valid(best_classes):
        # Fallback to a known valid simple solution
        return [[20 * j, j - 1] for j in range(1, 11)]
        
    return [[m, r] for m, r in best_classes]
