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
    
    # Sort moduli by 1/m descending (i.e., m ascending)
    # Actually, we want to pick small m first to maximize sum(1/m)
    # But we need to ensure disjointness.
    # Greedy: try to add moduli in order of increasing m (highest 1/m first).
    
    def is_valid_assignment(moduli, residues):
        """Check if the assignment (m_i, r_i) is pairwise disjoint."""
        n = len(moduli)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = moduli[i], residues[i]
                m2, r2 = moduli[j], residues[j]
                g = gcd(m1, m2)
                # Disjoint if r1 != r2 mod g
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
    
    # --- Phase 1: Greedy Construction ---
    # We process moduli from smallest to largest (highest 1/m first).
    # For each, we try to find a residue that doesn't conflict.
    
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
        # Fallback to seed if greedy failed (shouldn't happen)
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    best_moduli = selected_moduli[:]
    best_residues = selected_residues[:]
    best_score = current_score
    
    # --- Phase 2: Local Search ---
    # We will try to improve the set.
    # Moves:
    # 1. Add a missing modulus if possible.
    # 2. Swap a selected modulus for a missing one if it improves score.
    # 3. Reassign residues to unlock additions? (More complex, maybe skip for now or simple swap)
    
    missing_moduli = [m for m in all_moduli if m not in selected_moduli]
    # Sort missing by 1/m descending (m ascending)
    missing_moduli.sort()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding a missing modulus
        # We should try to add the one with highest 1/m that fits.
        # But adding might block future additions. Greedy add is a good heuristic.
        # Let's try to add any missing modulus that fits, prioritizing small m.
        
        added = False
        for idx_m, m in enumerate(missing_moduli):
            if time.time() > deadline:
                break
            r = find_valid_residue(m, best_moduli, best_residues)
            if r is not None:
                # Tentatively add
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
                    break  # Restart loop to re-evaluate with new state
        
        # 2. Try swapping
        # If no addition was possible, try swaps.
        # Or always try swaps after additions?
        # Let's try swaps if we are stuck or to improve further.
        
        if not added:
            # Try swapping a selected modulus (larger m, lower 1/m) with a missing one (smaller m, higher 1/m)
            # Sort selected by m descending (worst 1/m first)
            # Sort missing by m ascending (best 1/m first)
            
            # We need to check if removing a selected modulus allows adding a missing one,
            # and if the net score change is positive.
            # Also, the residue of the remaining moduli might need to change to allow the new one?
            # Simple swap: Remove (m_sel, r_sel), try to add (m_miss, r_miss) to the remaining set.
            
            # To be efficient, we can just try all pairs?
            # N_sel * N_miss could be large.
            # Let's iterate over missing (small m) and selected (large m).
            
            for m_miss in missing_moduli:
                if time.time() > deadline:
                    break
                for i, m_sel in enumerate(best_moduli):
                    # Only consider swap if m_miss < m_sel (improves 1/m sum directly)
                    if m_miss >= m_sel:
                        continue
                    
                    # Remove best_moduli[i]
                    temp_moduli = [m for j, m in enumerate(best_moduli) if j != i]
                    temp_residues = [r for j, r in enumerate(best_residues) if j != i]
                    
                    # Try to add m_miss
                    r_miss = find_valid_residue(m_miss, temp_moduli, temp_residues)
                    if r_miss is not None:
                        new_moduli = temp_moduli + [m_miss]
                        new_residues = temp_residues + [r_miss]
                        new_score = calc_score(new_moduli, new_residues)
                        if new_score is not None and new_score > best_score:
                            best_moduli = new_moduli
                            best_residues = new_residues
                            best_score = new_score
                            # Update missing list
                            if m_miss in missing_moduli:
                                missing_moduli.remove(m_miss)
                            if m_sel not in missing_moduli:
                                missing_moduli.append(m_sel)
                            missing_moduli.sort()
                            improved = True
                            break
                if improved:
                    break
    
    # Final check
    if not is_valid_assignment(best_moduli, best_residues):
        # Fallback
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    return [[m, r] for m, r in zip(best_moduli, best_residues)]

# EVOLVE-BLOCK-END
