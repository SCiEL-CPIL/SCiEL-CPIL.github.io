import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    moduli_range = range(20, 201)
    
    def is_valid_addition(m, r, selected):
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            if (r % g) == (r2 % g):
                return False
        return True

    def find_valid_residue(m, selected):
        # Find any residue r in [0, m) that is valid
        for r in range(m):
            if is_valid_addition(m, r, selected):
                return r
        return None

    def get_score(selected):
        return sum(1.0 / m for m, _ in selected)

    # 1. Greedy Initialization
    # Sort moduli by weight (descending) i.e., m ascending
    # We try to add them in order of m
    selected = []
    for m in moduli_range:
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    current_score = get_score(selected)
    
    # 2. Local Search
    # We will try to improve the score by:
    # a) Adding a missing modulus if possible.
    # b) Swapping a selected modulus m_old with a missing modulus m_new 
    #    if m_new < m_old (higher weight) and a valid residue exists.
    # c) Changing residues of selected moduli to enable additions or swaps.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try additions
        for m in moduli_range:
            if any(m == sm for sm, _ in selected):
                continue
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
                current_score += 1.0 / m
                improved = True
                # Don't break, try to add more if possible
                if time.time() > deadline - 5:
                    break
        
        if time.time() > deadline - 5:
            break

        # Try swaps
        # We want to replace a larger m with a smaller m
        # Iterate over selected moduli
        # To avoid O(N^2) complexity issues in inner loops, we just check all pairs
        # N is at most 181, so N^2 is ~32k, which is fine.
        
        # Create a list of indices to swap from
        # We need to check if removing m_old and adding m_new is valid.
        
        # Optimization: Only try to swap if there is a smaller m available
        selected_moduli_set = {m for m, _ in selected}
        
        for i, (m_old, r_old) in enumerate(selected):
            if time.time() > deadline - 5:
                break
            
            # Find candidate m_new < m_old not in selected
            # To make it efficient, we can just iterate m_new from 20 to m_old-1
            # and check if it's in selected_moduli_set
            for m_new in range(20, m_old):
                if m_new in selected_moduli_set:
                    continue
                
                # Try to find a residue for m_new given selected without m_old
                # We need to check validity against all other selected moduli
                temp_selected = [selected[j] for j in range(len(selected)) if j != i]
                
                r_new = find_valid_residue(m_new, temp_selected)
                if r_new is not None:
                    # Perform swap
                    # Update selected
                    selected[i] = (m_new, r_new)
                    selected_moduli_set.remove(m_old)
                    selected_moduli_set.add(m_new)
                    
                    # Update score
                    current_score -= 1.0 / m_old
                    current_score += 1.0 / m_new
                    
                    improved = True
                    # Break inner loop to re-evaluate other swaps with new state?
                    # Or continue? Continuing might find multiple swaps in one pass.
                    # But changing selected[i] might affect other checks.
                    # It's safer to restart the swap loop or just mark improved and break out to re-run the while loop.
                    # Given the cost, breaking out of the inner loop and letting the outer loop handle it is safer.
                    break
            if improved:
                break
        
        # Try residue changes to enable additions?
        # This is more complex. A simple heuristic:
        # For each selected (m, r), try changing r to see if it allows adding a missing m.
        # This is O(N * M * M). Might be too slow if N, M are large.
        # Let's stick to Add and Swap first. If score stagnates, we can add residue perturbation.
        
        if not improved:
            # If no add or swap worked, try random residue perturbations to escape local minima
            # Pick a random selected modulus and change its residue
            if selected:
                idx = random.randint(0, len(selected) - 1)
                m, r = selected[idx]
                new_r = random.randint(0, m - 1)
                if new_r != r:
                    # Check if new_r is valid against others
                    if is_valid_addition(m, new_r, [selected[j] for j in range(len(selected)) if j != idx]):
                        selected[idx] = (m, new_r)
                        improved = True # Force another iteration to check for adds/swaps
                        # Note: Score doesn't change, but structure does.
    
    # Final validation and formatting
    # Convert to [[m, r], ...]
    result = [[m, r] for m, r in selected]
    
    # Double check validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if (r1 % g) == (r2 % g):
                # Invalid, should not happen
                pass
    
    return result

# EVOLVE-BLOCK-END
