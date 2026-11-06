import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin before 60s
    
    # Candidate moduli: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    n_mods = len(moduli)
    
    # Precompute GCDs
    gcd_table = {}
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            gcd_table[(i, j)] = math.gcd(moduli[i], moduli[j])
    
    def is_compatible(r_new, m_idx_new, assigned):
        """
        assigned: dict {mod_idx: residue}
        Checks if (r_new, moduli[m_idx_new]) is compatible with all in assigned.
        """
        for m_idx_old, r_old in assigned.items():
            if m_idx_old == m_idx_new:
                continue
            g = gcd_table.get((min(m_idx_new, m_idx_old), max(m_idx_new, m_idx_old)))
            if (r_new - r_old) % g == 0:
                return False
        return True

    best_count = 0
    best_solution = {} # {mod_idx: residue}
    
    # Helper to convert dict to list of lists
    def to_output(assigned_dict):
        return [[assigned_dict[i], moduli[i]] for i in assigned_dict.keys()]
    
    # 1. Initial Greedy Construction
    # Sort moduli by "constraint tightness" - higher moduli have more residues, 
    # but lower moduli (like 2, 4) are more restrictive in terms of gcd?
    # Actually, for even moduli, gcd(m1, m2) is at least 2.
    # Smaller moduli have fewer possible residues, so they are "harder" to place?
    # Let's try sorting by modulus size (ascending) for greedy, as they have fewer options.
    order = sorted(range(n_mods), key=lambda i: moduli[i])
    
    # Try a few random greedy orders to find a good initial solution
    for _ in range(10):
        random.shuffle(order)
        current = {}
        for i in order:
            found = False
            # Try residues in random order
            residues = list(range(moduli[i]))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(r, i, current):
                    current[i] = r
                    found = True
                    break
            if not found:
                break # Can't place this modulus, move to next greedy attempt
        
        if len(current) > best_count:
            best_count = len(current)
            best_solution = current.copy()
            
    # 2. Local Search / Hill Climbing
    # We try to add unassigned moduli by swapping residues of assigned ones.
    
    # Precompute list of indices for random selection
    all_indices = list(range(n_mods))
    
    while time.time() - start_time < max_time:
        # Check if we have a perfect solution
        if best_count == n_mods:
            break
            
        # Pick a random unassigned modulus
        unassigned = [i for i in all_indices if i not in best_solution]
        if not unassigned:
            break
            
        m_idx_add = random.choice(unassigned)
        m_add = moduli[m_idx_add]
        
        added = False
        
        # Try all residues for the new modulus
        # Optimization: If we fail with random residues, try all? 
        # m_add is at most 128, so trying all is cheap.
        for r_add in range(m_add):
            # Check conflicts
            conflicts = []
            for m_idx_old, r_old in best_solution.items():
                g = gcd_table.get((min(m_idx_add, m_idx_old), max(m_idx_add, m_idx_old)))
                if (r_add - r_old) % g == 0:
                    conflicts.append((m_idx_old, r_old))
            
            if not conflicts:
                # Direct addition
                best_solution[m_idx_add] = r_add
                best_count += 1
                added = True
                break
            
            # If conflicts, try to resolve by swapping a conflicting modulus's residue
            # We only need to change ONE conflicting modulus's residue to resolve the conflict with r_add?
            # No, changing one might create new conflicts with others.
            # Let's try to change each conflicting modulus one by one.
            
            for m_idx_conf, r_conf in conflicts:
                # Try to find a new residue r_new_conf for m_idx_conf
                # such that:
                # 1. r_new_conf is compatible with all OTHER assigned moduli (excluding m_idx_conf)
                # 2. r_add is compatible with (m_idx_conf, r_new_conf)
                
                m_conf = moduli[m_idx_conf]
                # We can restrict search for r_new_conf?
                # Just try all residues for m_conf (max 128).
                
                # Optimization: Pre-check if r_add is compatible with m_idx_conf for ANY residue?
                # gcd(m_add, m_conf) = g. We need (r_add - r_new_conf) % g != 0.
                # This is true for (g-1)/g of residues. So usually possible.
                
                # Try a few random residues first, then all if needed?
                # Since m_conf <= 128, trying all is fast enough.
                for r_new_conf in range(m_conf):
                    # Condition 2: Compatible with new modulus
                    g_add_conf = gcd_table.get((min(m_idx_add, m_idx_conf), max(m_idx_add, m_idx_conf)))
                    if (r_add - r_new_conf) % g_add_conf == 0:
                        continue
                    
                    # Condition 1: Compatible with all other assigned moduli
                    valid_swap = True
                    for m_idx_other, r_other in best_solution.items():
                        if m_idx_other == m_idx_conf:
                            continue
                        g_other = gcd_table.get((min(m_idx_conf, m_idx_other), max(m_idx_conf, m_idx_other)))
                        if (r_new_conf - r_other) % g_other == 0:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        best_solution[m_idx_conf] = r_new_conf
                        best_solution[m_idx_add] = r_add
                        best_count += 1
                        added = True
                        break
                
                if added:
                    break
            
        if not added:
            # Local optimum reached for this unassigned modulus?
            # Try a random restart/perturbation: remove a random assigned modulus to break deadlock
            if best_solution:
                # Remove a random modulus
                to_remove = random.choice(list(best_solution.keys()))
                del best_solution[to_remove]
                best_count -= 1
                # Optionally, try to re-add it later? The loop will naturally try to add unassigned ones.
    
    return to_output(best_solution)
# EVOLVE-BLOCK-END
