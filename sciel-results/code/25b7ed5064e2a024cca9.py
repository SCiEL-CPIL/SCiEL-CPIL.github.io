"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli in order of 1/m (descending), adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) if the swap results in a valid assignment and higher total score.
       c. Reassigning residues to allow more moduli to be added.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Valid moduli range
    min_m, max_m = 20, 200
    all_moduli = list(range(min_m, max_m + 1))
    
    def is_valid_assignment(moduli, residues):
        """Check if the assignment (m_i, r_i) is pairwise disjoint."""
        n = len(moduli)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = moduli[i], residues[i]
                m2, r2 = moduli[j], residues[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    def find_valid_residue(m, existing_moduli, existing_residues):
        """
        Find a residue r in [0, m) such that (m, r) is disjoint from all existing pairs.
        Returns r or None.
        """
        for r in range(m):
            valid = True
            for i, (m_i, r_i) in enumerate(zip(existing_moduli, existing_residues)):
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def calc_score(moduli, residues):
        """Calculate the score sum(1/m)."""
        if not is_valid_assignment(moduli, residues):
            return None
        return sum(1.0 / m for m in moduli)
    
    def try_add_moduli_to_set(base_moduli, base_residues, candidates, time_limit):
        """
        Try to add as many candidate moduli as possible to the base set.
        Returns (new_moduli, new_residues, added_count).
        """
        cur_moduli = list(base_moduli)
        cur_residues = list(base_residues)
        added = 0
        remaining = list(candidates)
        
        # Sort by 1/m descending (m ascending)
        remaining.sort()
        
        while remaining and time.time() < time_limit:
            m = remaining.pop(0)
            r = find_valid_residue(m, cur_moduli, cur_residues)
            if r is not None:
                cur_moduli.append(m)
                cur_residues.append(r)
                added += 1
        
        return cur_moduli, cur_residues, added
    
    # --- Phase 1: Greedy Construction ---
    selected_moduli = []
    selected_residues = []
    
    for m in all_moduli:
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected_moduli, selected_residues)
        if r is not None:
            selected_moduli.append(m)
            selected_residues.append(r)
    
    current_score = calc_score(selected_moduli, selected_residues)
    if current_score is None:
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    best_moduli = selected_moduli[:]
    best_residues = selected_residues[:]
    best_score = current_score
    
    # --- Phase 2: Local Search ---
    missing_moduli = [m for m in all_moduli if m not in selected_moduli]
    missing_moduli.sort()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding a missing modulus
        added = False
        for idx_m, m in enumerate(missing_moduli):
            if time.time() > deadline:
                break
            r = find_valid_residue(m, best_moduli, best_residues)
            if r is not None:
                new_moduli = best_moduli + [m]
                new_residues = best_residues + [r]
                new_score = calc_score(new_moduli, new_residues)
                if new_score is not None and new_score > best_score:
                    best_moduli = new_moduli
                    best_residues = new_residues
                    best_score = new_score
                    missing_moduli.pop(idx_m)
                    improved = True
                    added = True
                    break
        
        # 2. Try swaps: remove one selected, add one missing
        if not added:
            # Sort selected by m descending (worst 1/m first)
            # Sort missing by m ascending (best 1/m first)
            sel_sorted = sorted(range(len(best_moduli)), key=lambda i: -best_moduli[i])
            
            for m_miss in missing_moduli:
                if time.time() > deadline:
                    break
                for i in sel_sorted:
                    m_sel = best_moduli[i]
                    if m_miss >= m_sel:
                        continue
                    
                    temp_moduli = [m for j, m in enumerate(best_moduli) if j != i]
                    temp_residues = [r for j, r in enumerate(best_residues) if j != i]
                    
                    r_miss = find_valid_residue(m_miss, temp_moduli, temp_residues)
                    if r_miss is not None:
                        new_moduli = temp_moduli + [m_miss]
                        new_residues = temp_residues + [r_miss]
                        new_score = calc_score(new_moduli, new_residues)
                        if new_score is not None and new_score > best_score:
                            best_moduli = new_moduli
                            best_residues = new_residues
                            best_score = new_score
                            if m_miss in missing_moduli:
                                missing_moduli.remove(m_miss)
                            if m_sel not in missing_moduli:
                                missing_moduli.append(m_sel)
                            missing_moduli.sort()
                            improved = True
                            break
                if improved:
                    break
        
        # 3. Try double swaps: remove two selected, add two missing
        if not added and time.time() < deadline:
            # Only try if we have enough missing and selected
            if len(missing_moduli) >= 2 and len(best_moduli) >= 2:
                # Get top missing by 1/m
                top_missing = missing_moduli[:min(5, len(missing_moduli))]
                # Get worst selected by 1/m
                sel_indices = sorted(range(len(best_moduli)), key=lambda i: -best_moduli[i])
                worst_sel = [best_moduli[i] for i in sel_indices[:min(5, len(sel_indices))]]
                
                for m_miss1 in top_missing:
                    if time.time() > deadline:
                        break
                    for m_miss2 in top_missing:
                        if m_miss2 <= m_miss1:
                            continue
                        # Find which selected to remove
                        for i1 in range(len(best_moduli)):
                            if time.time() > deadline:
                                break
                            for i2 in range(i1 + 1, len(best_moduli)):
                                m_sel1 = best_moduli[i1]
                                m_sel2 = best_moduli[i2]
                                # Only consider if both missing are smaller
                                if m_miss1 >= m_sel1 or m_miss2 >= m_sel2:
                                    continue
                                # Remove both selected
                                temp_moduli = [m for j, m in enumerate(best_moduli) if j != i1 and j != i2]
                                temp_residues = [r for j, r in enumerate(best_residues) if j != i1 and j != i2]
                                
                                # Try to add both missing
                                r_miss1 = find_valid_residue(m_miss1, temp_moduli, temp_residues)
                                if r_miss1 is None:
                                    continue
                                temp_moduli2 = temp_moduli + [m_miss1]
                                temp_residues2 = temp_residues + [r_miss1]
                                r_miss2 = find_valid_residue(m_miss2, temp_moduli2, temp_residues2)
                                if r_miss2 is None:
                                    continue
                                new_moduli = temp_moduli2 + [m_miss2]
                                new_residues = temp_residues2 + [r_miss2]
                                new_score = calc_score(new_moduli, new_residues)
                                if new_score is not None and new_score > best_score:
                                    best_moduli = new_moduli
                                    best_residues = new_residues
                                    best_score = new_score
                                    if m_miss1 in missing_moduli:
                                        missing_moduli.remove(m_miss1)
                                    if m_miss2 in missing_moduli:
                                        missing_moduli.remove(m_miss2)
                                    if m_sel1 not in missing_moduli:
                                        missing_moduli.append(m_sel1)
                                    if m_sel2 not in missing_moduli:
                                        missing_moduli.append(m_sel2)
                                    missing_moduli.sort()
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # 4. Residue reassignment: try to find better residues for existing moduli
        # This can unlock additions that were previously blocked
        if not added and time.time() < deadline:
            # Try reassigning residues one at a time
            reassigned = False
            for i in range(len(best_moduli)):
                if time.time() > deadline:
                    break
                m_i = best_moduli[i]
                # Try different residues for this modulus
                for r_new in range(m_i):
                    if r_new == best_residues[i]:
                        continue
                    temp_residues = best_residues[:]
                    temp_residues[i] = r_new
                    if is_valid_assignment(best_moduli, temp_residues):
                        # Check if this allows adding a missing modulus
                        for m_miss in missing_moduli[:10]:
                            r_miss = find_valid_residue(m_miss, best_moduli, temp_residues)
                            if r_miss is not None:
                                new_moduli = best_moduli + [m_miss]
                                new_residues = temp_residues + [r_miss]
                                new_score = calc_score(new_moduli, new_residues)
                                if new_score is not None and new_score > best_score:
                                    best_moduli = new_moduli
                                    best_residues = new_residues
                                    best_score = new_score
                                    if m_miss in missing_moduli:
                                        missing_moduli.remove(m_miss)
                                    missing_moduli.sort()
                                    improved = True
                                    reassigned = True
                                    break
                    if reassigned:
                        break
                if reassigned:
                    break
    
    # Final check
    if not is_valid_assignment(best_moduli, best_residues):
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    return [[m, r] for m, r in zip(best_moduli, best_residues)]

# EVOLVE-BLOCK-END
