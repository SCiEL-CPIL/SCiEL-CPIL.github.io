import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table and index map
    n_mod = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    best_result = [[1, 2]]  # fallback: at least one valid class
    best_count = 1
    
    def try_add_residue(m_idx, selected):
        """Find a residue for modulus moduli[m_idx] disjoint from all selected."""
        m = moduli[m_idx]
        # Precompute constraints: (g, r_i) for each selected where gcd > 1
        constraints = []
        for (r_i, m_i) in selected:
            m_i_idx = mod_to_idx[m_i]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))
        if not constraints:
            return 0
        # Try residues in random order for diversity
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy_randomized():
        """One greedy pass with random ordering."""
        selected = []  # list of [residue, modulus]
        used_moduli = set()
        order = list(range(n_mod))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            m = moduli[idx]
            if m in used_moduli:
                continue
            # Check gcd > 1 with all selected moduli
            compatible = True
            for (r_i, m_i) in selected:
                m_i_idx = mod_to_idx[m_i]
                if gcd_table[idx][m_i_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            # Try to find a valid residue
            r = try_add_residue(idx, selected)
            if r is not None:
                selected.append([r, m])
                used_moduli.add(m)
        
        return selected
    
    def local_search_improve(current):
        """Try to improve by adding, removing, or swapping moduli."""
        current = [list(x) for x in current]
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            used = set(m for _, m in current)
            
            # Phase 1: Try adding moduli
            for idx in range(n_mod):
                if time.time() > deadline - 3.0:
                    break
                m = moduli[idx]
                if m in used:
                    continue
                compatible = True
                for (r_i, m_i) in current:
                    m_i_idx = mod_to_idx[m_i]
                    if gcd_table[idx][m_i_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = try_add_residue(idx, current)
                if r is not None:
                    current.append([r, m])
                    used.add(m)
                    improved = True
            
            # Phase 2: Try removing one modulus and re-adding with different residue
            if not improved:
                for rem_idx in range(len(current)):
                    if time.time() > deadline - 3.0:
                        break
                    saved = current.pop(rem_idx)
                    saved_m = saved[1]
                    used.discard(saved_m)
                    
                    # Try adding other moduli that were previously blocked
                    added_any = False
                    for add_idx in range(n_mod):
                        if time.time() > deadline - 3.0:
                            break
                        m = moduli[add_idx]
                        if m in used or m == saved_m:
                            continue
                        compatible = True
                        for (r_i, m_i) in current:
                            m_i_idx = mod_to_idx[m_i]
                            if gcd_table[add_idx][m_i_idx] == 1:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        r = try_add_residue(add_idx, current)
                        if r is not None:
                            current.append([r, m])
                            used.add(m)
                            added_any = True
                            break  # only add one at a time to keep search tractable
                    
                    # Try re-adding the removed modulus with a different residue
                    r = try_add_residue(mod_to_idx[saved_m], current)
                    if r is not None:
                        current.append([r, saved_m])
                        used.add(saved_m)
                    else:
                        # If we can't re-add, only accept if we gained something
                        if not added_any:
                            current.append(saved)
                            used.add(saved_m)
                            continue
                    
                    if len(current) > rem_idx + 1 or added_any:
                        improved = True
                        break  # restart the while loop with new state
        
        return current
    
    # Run multiple randomized greedy passes
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        result = greedy_randomized()
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Try to improve best_result with local search
    if time.time() < deadline - 5.0:
        improved_result = local_search_improve(best_result)
        if len(improved_result) > best_count:
            best_count = len(improved_result)
            best_result = improved_result
    
    # Final attempt: more restarts if time remains
    if time.time() < deadline - 2.0:
        for restart in range(500):
            if time.time() > deadline - 1.0:
                break
            result = greedy_randomized()
            if len(result) > best_count:
                best_count = len(result)
                best_result = result
                # Quick local search on the new best
                if time.time() < deadline - 2.0:
                    improved_result = local_search_improve(best_result)
                    if len(improved_result) > best_count:
                        best_count = len(improved_result)
                        best_result = improved_result
    
    return best_result
