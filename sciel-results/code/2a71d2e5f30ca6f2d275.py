import random
from math import gcd

def construct():
    # Initial greedy solution
    selected = []
    for m in range(20, 201):
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
    
    # Local search improvement
    # We want to maximize sum(1/m)
    # Moves:
    # 1. Try to change residue of an existing modulus to allow adding a new one?
    #    Actually, changing residue doesn't change the score (1/m is fixed for m).
    #    But it might allow adding more moduli.
    # 2. Try to replace a large modulus with a smaller one?
    #    If we remove a large m, we might be able to add a smaller m' that was previously blocked.
    
    # Let's implement a simple hill climbing:
    # 1. Try to add any unselected m. If it fits, add it.
    # 2. If no m fits, try to remove a selected m (largest first) and see if we can add two or more smaller ones?
    #    Or just try to add a smaller m by removing a larger one?
    
    # Better: Simulated Annealing or Random Restart.
    # Given 120s, we can do many iterations.
    
    best_selected = [tuple(x) for x in selected]
    best_score = sum(1.0/m for m, r in best_selected)
    
    unselected = [m for m in range(20, 201) if not any(m == m_i for m_i, r_i in best_selected)]
    
    # Helper to check if a set is valid
    def is_valid(classes):
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
        
    # Helper to get score
    def get_score(classes):
        return sum(1.0/m for m, r in classes)

    # Try to improve by swapping
    # Since residue choice affects validity, we need to be careful.
    # For a given set of moduli, is there always a valid assignment of residues?
    # Not necessarily. But if we start with a valid assignment, we can keep it.
    # If we change the set of moduli, we need to find valid residues.
    # We can use the greedy residue assignment for the new set.
    
    def greedy_residues(moduli):
        res = []
        for m in moduli:
            found = False
            for r in range(m):
                valid = True
                for (m_i, r_i) in res:
                    g = gcd(m, m_i)
                    if r % g == r_i % g:
                        valid = False
                        break
                if valid:
                    res.append((m, r))
                    found = True
                    break
            if not found:
                return None # No valid assignment for this order
        return res

    # We will try to replace a subset of moduli with a different subset.
    # Simple move: Remove one modulus, add one modulus.
    # To allow adding a smaller modulus, we might need to remove a larger one.
    
    improved = True
    while improved:
        improved = False
        current_moduli = [m for m, r in best_selected]
        
        # Try to add a new modulus
        for m_new in unselected:
            # Try to add m_new to current set
            # We need to check if there exists a residue for m_new
            # compatible with all current residues.
            # We can just try all r for m_new.
            added = False
            for r_new in range(m_new):
                valid = True
                for (m_i, r_i) in best_selected:
                    g = gcd(m_new, m_i)
                    if r_new % g == r_i % g:
                        valid = False
                        break
                if valid:
                    # Check if the new set is valid (it is, by construction)
                    new_selected = best_selected + [(m_new, r_new)]
                    new_score = get_score(new_selected)
                    if new_score > best_score:
                        best_selected = new_selected
                        best_score = new_score
                        unselected.remove(m_new)
                        improved = True
                        added = True
                        break
            if added:
                break # Restart loop with new best
                
        # If no addition, try to swap
        # Remove a modulus m_old and add m_new (m_new < m_old)
        # This is complex. Let's stick to additions first.
        # If stuck, try random restarts or different initial greedy order.
        
    return [[m, r] for (m, r) in best_selected]
