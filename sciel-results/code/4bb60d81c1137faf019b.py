import math
import time
import random

def construct():
    """
    Constructs a set of pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Returns:
        list of [m, r] pairs
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute GCD table for moduli 20..200
    moduli = list(range(20, 201))
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def _is_compatible(m, r, selected):
        """Check if (m, r) is compatible with all entries in selected."""
        for m_prime, r_prime in selected:
            g = gcd_table.get((m, m_prime), math.gcd(m, m_prime))
            if r % g == r_prime % g:
                return False
        return True
    
    def _score(selected):
        """Compute the score of a selection."""
        return sum(1.0 / m for m, r in selected)
    
    def _greedy_construct(order, deadline):
        """Greedy construction using the given order of moduli."""
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            # Try residues in order 0, 1, ..., m-1
            for r in range(m):
                if _is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def _local_improve(selected, deadline):
        """
        Optimized local improvement:
        1. Try to add any missing modulus that fits directly.
        2. If no direct fit, try simple swaps (change residue of one to allow another).
        """
        selected_moduli = set(m for m, r in selected)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add missing moduli directly
            # Sort missing moduli by value (smaller moduli give higher 1/m, so prioritize them? 
            # Actually, we just want ANY valid addition. Smaller moduli are harder to fit 
            # because they have more constraints relative to their size? 
            # Let's just iterate through all missing moduli.
            missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
            
            for m_new in missing_moduli:
                if time.time() > deadline:
                    break
                    
                # Find a valid residue for m_new
                valid_r = None
                for r_new in range(m_new):
                    if _is_compatible(m_new, r_new, selected):
                        valid_r = r_new
                        break
                
                if valid_r is not None:
                    selected.append([m_new, valid_r])
                    selected_moduli.add(m_new)
                    improved = True
                    # Restart the loop to check for more additions
                    break
            
            # Phase 2: If no direct addition, try swapping residues
            if not improved:
                # Try to change the residue of an existing element to free up space for a missing one
                # We limit the scope of this search to keep it fast.
                # Only try to swap if we have a specific target missing modulus in mind.
                # Pick the smallest missing modulus as a target (highest potential gain).
                if missing_moduli:
                    target_m = min(missing_moduli)
                    
                    for i in range(len(selected)):
                        if time.time() > deadline:
                            break
                            
                        m_i, r_i = selected[i]
                        old_r = r_i
                        
                        # Try changing r_i to a different value
                        for r_alt in range(m_i):
                            if r_alt == r_i:
                                continue
                            
                            # Temporarily set r_alt
                            selected[i][1] = r_alt
                            
                            # Check if target_m can now be added
                            can_add_target = False
                            for r_new in range(target_m):
                                if _is_compatible(target_m, r_new, selected):
                                    can_add_target = True
                                    break
                            
                            if can_add_target:
                                # Add target
                                selected.append([target_m, r_new])
                                selected_moduli.add(target_m)
                                improved = True
                                break
                            else:
                                # Revert
                                selected[i][1] = old_r
                        if improved:
                            break
                else:
                    # No missing moduli, we are done
                    break
                    
        return selected
    
    # Run multiple restarts with different orderings
    best_solution = []
    best_score = 0.0
    
    # Initial greedy with natural order (smallest first)
    moduli_natural = list(range(20, 201))
    sol1 = _greedy_construct(moduli_natural, deadline)
    sol1 = _local_improve(sol1, deadline)
    score1 = _score(sol1)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
    
    # Random restarts with shuffled orders
    # Reduced restart count to ensure we stay within time limits
    random.seed(42)
    
    # Estimate time per restart: Greedy is fast (<1s). Local improve is optimized.
    # Let's allow up to 20 restarts, checking time each time.
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        moduli_shuffled = list(range(20, 201))
        random.shuffle(moduli_shuffled)
        
        sol = _greedy_construct(moduli_shuffled, deadline - 1)
        
        # Local improvement if time permits
        if time.time() < deadline - 1:
            sol = _local_improve(sol, deadline - 0.5)
        
        score = _score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    return best_solution
