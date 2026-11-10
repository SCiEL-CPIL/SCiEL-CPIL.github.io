"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Greedy: try to include all moduli from 20 to 200
    # Process in order of increasing m (highest weight 1/m first)
    selected = []  # list of [m, r]
    
    for m in range(20, 201):
        # Check time limit
        if time.time() > deadline:
            break
        
        # Try each residue 0, 1, ..., m-1
        found = False
        for r in range(m):
            # Check compatibility with all selected classes
            compatible = True
            for m2, r2 in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        # If no compatible residue found, skip this modulus
    
    def score(sel):
        return sum(1.0/m for m, r in sel)
    
    current_score = score(selected)
    
    moduli_in_selected = set(m for m, r in selected)
    all_moduli = set(range(20, 201))
    missing_moduli = sorted(all_moduli - moduli_in_selected)
    
    # Phase 1: Single-add and single-swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m_new in missing_moduli:
            if time.time() > deadline:
                break
            
            for r_new in range(m_new):
                conflicts = []
                for idx, (m2, r2) in enumerate(selected):
                    g = gcd(m_new, m2)
                    if r_new % g == r2 % g:
                        conflicts.append(idx)
                
                if len(conflicts) == 0:
                    selected.append([m_new, r_new])
                    current_score += 1.0/m_new
                    missing_moduli = [x for x in missing_moduli if x != m_new]
                    improved = True
                    break
                elif len(conflicts) == 1:
                    idx_remove = conflicts[0]
                    m_remove, r_remove = selected[idx_remove]
                    if 1.0/m_new > 1.0/m_remove:
                        selected.pop(idx_remove)
                        selected.append([m_new, r_new])
                        current_score = current_score - 1.0/m_remove + 1.0/m_new
                        missing_moduli = [x for x in missing_moduli if x != m_new]
                        improved = True
                        break
            
            if improved:
                break
    
    # Phase 2: Residue perturbation - try changing residues of existing elements
    # to free up space for missing moduli
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved_this_round = False
        
        # Try reassigning residues of existing elements
        for idx in range(len(selected)):
            if time.time() > deadline:
                break
            
            m_cur, r_cur = selected[idx]
            # Try different residues for this element
            for r_try in range(m_cur):
                if r_try == r_cur:
                    continue
                
                # Check if new residue is compatible with all other selected elements
                compatible = True
                for jdx, (m2, r2) in enumerate(selected):
                    if jdx == idx:
                        continue
                    g = gcd(m_cur, m2)
                    if r_try % g == r2 % g:
                        compatible = False
                        break
                
                if compatible:
                    # Check if this allows adding a missing modulus
                    for m_new in missing_moduli:
                        if time.time() > deadline:
                            break
                        
                        # Find a residue for m_new compatible with the modified set
                        for r_new in range(m_new):
                            ok = True
                            for jdx, (m2, r2) in enumerate(selected):
                                if jdx == idx:
                                    # Use the proposed new residue
                                    if r_new % gcd(m_new, m_cur) == r_try % gcd(m_new, m_cur):
                                        ok = False
                                        break
                                else:
                                    if r_new % gcd(m_new, m2) == r2 % gcd(m_new, m2):
                                        ok = False
                                        break
                            
                            if ok:
                                # Apply both changes
                                selected[idx][1] = r_try
                                selected.append([m_new, r_new])
                                current_score += 1.0/m_new
                                missing_moduli = [x for x in missing_moduli if x != m_new]
                                improved_this_round = True
                                break
                        
                        if improved_this_round:
                            break
                
                if improved_this_round:
                    break
            
            if improved_this_round:
                break
        
        if not improved_this_round:
            break
    
    # Phase 3: 2-opt swap - remove two elements and try to add two missing ones
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved_this_round = False
        
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            for j in range(i+1, len(selected)):
                if time.time() > deadline:
                    break
                
                m_i, r_i = selected[i]
                m_j, r_j = selected[j]
                
                # Try removing both and adding two missing moduli
                for m_new1 in missing_moduli:
                    if time.time() > deadline:
                        break
                    
                    for m_new2 in missing_moduli:
                        if m_new2 <= m_new1:
                            continue
                        
                        if 1.0/m_new1 + 1.0/m_new2 <= 1.0/m_i + 1.0/m_j:
                            continue
                        
                        # Try to find residues for both new moduli
                        # that are compatible with the remaining selected set
                        remaining = []
                        for idx, (m2, r2) in enumerate(selected):
                            if idx != i and idx != j:
                                remaining.append((m2, r2))
                        
                        found_r1 = False
                        for r1 in range(m_new1):
                            # Check compatibility with remaining
                            ok1 = True
                            for m2, r2 in remaining:
                                if r1 % gcd(m_new1, m2) == r2 % gcd(m_new1, m2):
                                    ok1 = False
                                    break
                            
                            if not ok1:
                                continue
                            
                            for r2 in range(m_new2):
                                # Check compatibility with remaining and with (m_new1, r1)
                                ok2 = True
                                for m3, r3 in remaining:
                                    if r2 % gcd(m_new2, m3) == r3 % gcd(m_new2, m3):
                                        ok2 = False
                                        break
                                
                                if not ok2:
                                    continue
                                
                                # Check compatibility between the two new ones
                                g12 = gcd(m_new1, m_new2)
                                if r1 % g12 == r2 % g12:
                                    continue
                                
                                # Apply the swap
                                selected[i] = [m_new1, r1]
                                selected[j] = [m_new2, r2]
                                current_score = current_score - 1.0/m_i - 1.0/m_j + 1.0/m_new1 + 1.0/m_new2
                                missing_moduli = [x for x in missing_moduli if x != m_new1 and x != m_new2]
                                improved_this_round = True
                                break
                            
                            if improved_this_round:
                                break
                        
                        if improved_this_round:
                            break
                    
                    if improved_this_round:
                        break
                
                if improved_this_round:
                    break
            
            if improved_this_round:
                break
        
        if not improved_this_round:
            break
    
    return selected
# EVOLVE-BLOCK-END
