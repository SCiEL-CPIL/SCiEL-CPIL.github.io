import math
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def is_compatible_with_list(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible(m, r, m2, r2):
                return False
        return True

    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    def find_valid_residue(m, sel):
        # Returns a valid residue for m given sel, or None
        # Shuffling residues helps find a solution if one exists
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible_with_list(m, r, sel):
                return r
        return None

    moduli = list(range(20, 201))
    
    best_selected = []
    best_score = 0.0

    # 1. Greedy Initialization with multiple restarts
    # We process moduli in ascending order (highest 1/m first).
    # Randomizing the residue selection order in each restart explores different local optima.
    for _ in range(20):
        selected = []
        for m in moduli:
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = list(selected)

    # 2. Local Search
    current = list(best_selected)
    current_score = best_score
    used_moduli = set(m for m, r in current)
    skipped = [m for m in moduli if m not in used_moduli]
    
    # Precompute gcd matrix? 
    # gcd is fast enough.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Add any compatible skipped modulus
        # We iterate through skipped list. If we add one, we restart the check for others?
        # Or just continue? 
        # Since adding one might block another, we should be careful.
        # But if we add, score increases. We can just greedily add all possible.
        new_skipped = []
        for m in skipped:
            r = find_valid_residue(m, current)
            if r is not None:
                current.append((m, r))
                used_moduli.add(m)
                current_score += 1.0 / m
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
            else:
                new_skipped.append(m)
        skipped = new_skipped
        
        # Phase 2: 2-Opt Swap
        # Try to replace a selected modulus with a skipped one if it increases score.
        # We only do this if we didn't add anything in Phase 1 (or even if we did, 
        # but usually we want to fill up first).
        # Let's do it always if there are skipped moduli.
        if skipped and current:
            # To avoid O(N*S*N) complexity being too high, we limit attempts or optimize.
            # N ~ 80, S ~ 100.
            # For each m_out in current:
            #   For each m_in in skipped:
            #     Check if m_in can be added to current \ {m_out}
            # This is 80 * 100 * 80 = 640,000 compatibility checks.
            # Each check is O(1) if we precompute? No, O(1) gcd + mod.
            # 640k ops is < 1ms in Python? Maybe 10ms.
            # So we can do this full search for swaps every iteration.
            
            swap_improved = False
            # We need to be careful with indices if we modify the list.
            # We will identify the best swap first, then apply it.
            best_swap_score = -1
            best_swap_idx = -1
            best_swap_m_in = -1
            best_swap_r_in = -1
            
            for i, (m_out, r_out) in enumerate(current):
                # Create a view of current without m_out
                # Instead of slicing, we can just skip index i in compatibility check
                for m_in in skipped:
                    if m_in == m_out:
                        continue
                    # Check if m_in has a valid residue in current \ {m_out}
                    # We need to find *any* valid residue.
                    # find_valid_residue is O(m * N).
                    # This is too slow: 80 * 100 * 100 * 80 = 64,000,000 ops.
                    # Too slow for Python in a tight loop.
                    
                    # Optimization: 
                    # We can check compatibility more cheaply?
                    # Or we can just try a few random residues for m_in?
                    # If a valid residue exists, random search will likely find it quickly
                    # if the conflict graph is not too dense.
                    # Let's try up to 5 random residues for m_in.
                    
                    found_r = None
                    # Try 5 random residues
                    for _ in range(5):
                        r_test = random.randint(0, m_in - 1)
                        compatible = True
                        for j, (m2, r2) in enumerate(current):
                            if j == i:
                                continue
                            if not is_compatible(m_in, r_test, m2, r2):
                                compatible = False
                                break
                        if compatible:
                            found_r = r_test
                            break
                    
                    if found_r is not None:
                        new_score = current_score - 1.0 / m_out + 1.0 / m_in
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap_idx = i
                            best_swap_m_in = m_in
                            best_swap_r_in = found_r
            
            if best_swap_idx != -1 and best_swap_score > current_score + 1e-12:
                # Apply swap
                m_out, r_out = current[best_swap_idx]
                current[best_swap_idx] = (best_swap_m_in, best_swap_r_in)
                used_moduli.remove(m_out)
                used_moduli.add(best_swap_m_in)
                skipped.remove(best_swap_m_in)
                skipped.append(m_out)
                current_score = best_swap_score
                best_score = current_score
                best_selected = list(current)
                improved = True
                swap_improved = True

        # Phase 3: Residue Perturbation
        # If no swap found, try changing a residue of an existing modulus
        # to see if it allows adding a new one.
        if not improved and current and skipped:
            # Pick a random modulus
            idx = random.randint(0, len(current) - 1)
            m, r_old = current[idx]
            # Try a few new residues
            for _ in range(10):
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                # Check compatibility with others
                compatible = True
                for j, (m2, r2) in enumerate(current):
                    if j == idx:
                        continue
                    if not is_compatible(m, r_new, m2, r2):
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # If compatible, try to add a skipped modulus
                current[idx] = (m, r_new)
                # Try to add any skipped modulus
                added = False
                for m_add in skipped:
                    r_add = find_valid_residue(m_add, current)
                    if r_add is not None:
                        current.append((m_add, r_add))
                        new_score = current_score + 1.0 / m_add
                        if new_score > best_score:
                            used_moduli.add(m_add)
                            skipped.remove(m_add)
                            current_score = new_score
                            best_score = new_score
                            best_selected = list(current)
                            improved = True
                            added = True
                        break
                
                if not added:
                    # Revert residue change
                    current[idx] = (m, r_old)
                else:
                    break

    return [[m, r] for m, r in best_selected]
