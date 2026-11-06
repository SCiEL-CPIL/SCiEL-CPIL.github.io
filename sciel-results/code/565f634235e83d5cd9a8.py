"""Combined program: greedy by value with local improvement passes."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START

def _is_compatible(m, r, selected):
    """Check if (m, r) is compatible with all entries in selected."""
    for (m2, r2) in selected:
        g = gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _find_best_residue(m, selected):
    """Find the residue r for modulus m that is compatible with selected, or None."""
    for r in range(m):
        if _is_compatible(m, r, selected):
            return r
    return None

def _greedy_construct(deadline):
    """Greedy construction: pick moduli in ascending order (highest 1/m first)."""
    moduli = list(range(20, 201))
    chosen = []
    
    for m in moduli:
        if time.time() > deadline:
            break
        
        # Find a compatible residue
        r = _find_best_residue(m, chosen)
        if r is not None:
            chosen.append([m, r])
    
    return chosen

def _local_improve(chosen, deadline):
    """Try to improve the solution by swapping residues or adding/removing moduli."""
    chosen = [list(c) for c in chosen]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any missing modulus
        current_moduli = set(m for m, r in chosen)
        for m in range(20, 201):
            if m in current_moduli:
                continue
            if time.time() > deadline:
                break
            
            r = _find_best_residue(m, chosen)
            if r is not None:
                chosen.append([m, r])
                current_moduli.add(m)
                improved = True
        
        # Try swapping residues to free up space for better moduli
        # For each pair of selected moduli, try to find alternative residues
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            m_i, r_i = chosen[i]
            # Try different residues for m_i
            for r_new in range(m_i):
                if r_new == r_i:
                    continue
                # Check if r_new is compatible with all other selected
                compatible = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    m_j, r_j = chosen[j]
                    g = gcd(m_i, m_j)
                    if r_new % g == r_j % g:
                        compatible = False
                        break
                if compatible:
                    # Check if this allows adding a new modulus
                    chosen[i][1] = r_new
                    current_moduli = set(m for m, r in chosen)
                    for m_new in range(20, 201):
                        if m_new in current_moduli:
                            continue
                        r_new2 = _find_best_residue(m_new, chosen)
                        if r_new2 is not None:
                            # This swap is beneficial
                            chosen.append([m_new, r_new2])
                            improved = True
                            break
                    if not improved:
                        # Revert the swap
                        chosen[i][1] = r_i
    
    return chosen

def _score(chosen):
    """Compute the score: sum of 1/m for all selected moduli."""
    return sum(1.0 / m for m, r in chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s limit
    
    # Phase 1: Greedy construction
    best = _greedy_construct(deadline)
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if _score(improved) > _score(best):
            best = improved
    
    # Phase 3: Try multiple random restarts with different orderings if time permits
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            # Shuffle moduli order
            moduli_shuffled = list(range(20, 201))
            random.shuffle(moduli_shuffled)
            
            chosen = []
            for m in moduli_shuffled:
                if time.time() > deadline - 2:
                    break
                r = _find_best_residue(m, chosen)
                if r is not None:
                    chosen.append([m, r])
            
            if _score(chosen) > _score(best):
                best = chosen
    
    return best

# EVOLVE-BLOCK-END
