"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find a valid residue for modulus m given the selected classes. Returns None if impossible."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def greedy_fill(selected):
        """Fill the selected list with the smallest possible moduli that fit."""
        # Convert to dict for fast lookup of existing moduli
        existing_moduli = {m for m, r in selected}
        
        # We want to add moduli in increasing order of m to maximize 1/m sum
        for m in range(20, 201):
            if m in existing_moduli:
                continue
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
                existing_moduli.add(m)
        return selected

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    # 1. Initial Greedy Construction
    selected = []
    selected = greedy_fill(selected)
    best_selected = [x[:] for x in selected]
    best_score = score(selected)
    
    if time.time() > deadline:
        return best_selected

    # 2. Local Search / Hill Climbing
    # We try to improve the set by adding moduli or swapping large moduli for small ones.
    
    while time.time() < deadline:
        improved = False
        
        # Step A: Try to Add any missing modulus
        # We prioritize small moduli (high 1/m)
        existing_moduli = {m for m, r in selected}
        candidates_to_add = [m for m in range(20, 201) if m not in existing_moduli]
        
        # Sort candidates by 1/m (descending) i.e., m ascending
        for m_in in candidates_to_add:
            r = find_valid_residue(m_in, selected)
            if r is not None:
                # Add it and re-fill
                selected.append([m_in, r])
                selected = greedy_fill(selected)
                new_score = score(selected)
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in selected]
                improved = True
                break # Restart the main loop with the new state
        
        if improved:
            continue

        # Step B: Try to Swap
        # Swap out a large modulus for a small unselected one
        # Sort selected by m descending (largest first, least valuable)
        selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
        
        # Candidates to swap in: unselected moduli, sorted ascending (smallest first, most valuable)
        existing_moduli = {m for m, r in selected}
        candidates_to_swap_in = [m for m in range(20, 201) if m not in existing_moduli]
        
        swap_performed = False
        for m_out in selected_sorted:
            m_out_val = m_out[0]
            for m_in in candidates_to_swap_in:
                # Only swap if the new modulus is significantly better (smaller)
                if 1.0 / m_in <= 1.0 / m_out_val:
                    continue
                
                # Tentatively remove m_out
                temp_selected = [x[:] for x in selected if x[0] != m_out_val]
                
                # Check if m_in can be added
                r = find_valid_residue(m_in, temp_selected)
                if r is not None:
                    temp_selected.append([m_in, r])
                    # Re-fill the gap left by m_out and potentially others
                    temp_selected = greedy_fill(temp_selected)
                    
                    new_score = score(temp_selected)
                    if new_score > best_score:
                        selected = temp_selected
                        best_score = new_score
                        best_selected = [x[:] for x in selected]
                        swap_performed = True
                        break
            
            if swap_performed:
                break
        
        if swap_performed:
            continue

        # Step C: Perturbation (if stuck)
        # Randomly remove a few moduli and re-greedy
        if len(selected) > 5:
            # Remove 3 random moduli, preferring larger ones
            indices_to_remove = random.sample(range(len(selected)), min(3, len(selected)))
            # Sort indices to remove, prefer larger moduli? 
            # Let's just remove random ones to explore landscape
            temp_selected = [x[:] for x in selected if selected.index(x) not in indices_to_remove]
            # The above index check is O(N^2). Better:
            temp_selected = []
            remove_set = set(indices_to_remove)
            for i, item in enumerate(selected):
                if i not in remove_set:
                    temp_selected.append(item[:])
            
            temp_selected = greedy_fill(temp_selected)
            new_score = score(temp_selected)
            if new_score > best_score:
                selected = temp_selected
                best_score = new_score
                best_selected = [x[:] for x in selected]
            else:
                # If perturbation didn't help, we might be stuck in a local optimum.
                # To avoid infinite loop on same perturbation, we can force a change
                # or just break if we've tried many times. 
                # Given the time limit, we can just let it run until deadline.
                pass

    return best_selected
# EVOLVE-BLOCK-END
