"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_disjoint_set(classes):
        """Check if all pairs are disjoint."""
        for i in range(len(classes)):
            m1, r1 = classes[i]
            for j in range(i + 1, len(classes)):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True

    def can_add(classes, m, r):
        """Check if (m, r) can be added to classes without conflict."""
        for m2, r2 in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def score(classes):
        """Calculate exact score as Fraction."""
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s

    def count_future_compatibility(current_set, m, r):
        """
        Count how many future moduli (not yet in current_set) could still be added
        after placing (m, r). Higher count = more flexible choice.
        """
        current_mods = {cm for cm, cr in current_set}
        count = 0
        for m2 in moduli:
            if m2 in current_mods or m2 == m:
                continue
            g = gcd(m, m2)
            # After adding (m, r), m2 is compatible if there exists r2 with
            # (r2 - r) % g != 0 AND (r2 - r_j) % gcd(m2, m_j) != 0 for all existing
            # Simplified: check if at least one residue for m2 avoids conflict with (m,r)
            # and with all existing classes
            # A residue r2 for m2 conflicts with (m,r) iff (r2 - r) % g == 0
            # Number of r2 in [0, m2) that conflict with (m,r): floor(m2/g) or ceil
            # We need at least one r2 that doesn't conflict with (m,r) AND doesn't conflict with existing
            # This is complex, so use a simpler heuristic:
            # Just check if there exists any r2 that works
            # For speed, sample a few residues
            found = False
            # Try up to 5 random residues
            for _ in range(5):
                r2 = random.randint(0, m2 - 1)
                if can_add(current_set + [[m, r]], m2, r2):
                    found = True
                    break
            if found:
                count += 1
        return count

    def pick_best_residue(current_set, m, future_moduli_set):
        """
        Pick the residue r for modulus m that maximizes future flexibility.
        For each valid r, count how many future moduli can still be placed.
        """
        valid_residues = []
        for r in range(m):
            if can_add(current_set, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        if len(valid_residues) <= 5:
            # If few choices, evaluate all
            best_r = None
            best_count = -1
            for r in valid_residues:
                temp_set = current_set + [[m, r]]
                count = 0
                for m2 in future_moduli_set:
                    if m2 == m:
                        continue
                    # Quick check: can m2 be added?
                    # Sample a few residues
                    for _ in range(3):
                        r2 = random.randint(0, m2 - 1)
                        if can_add(temp_set, m2, r2):
                            count += 1
                            break
                if count > best_count:
                    best_count = count
                    best_r = r
            return best_r
        else:
            # Many choices: sample a subset
            sample_size = min(20, len(valid_residues))
            sampled = random.sample(valid_residues, sample_size)
            best_r = None
            best_count = -1
            for r in sampled:
                temp_set = current_set + [[m, r]]
                count = 0
                for m2 in list(future_moduli_set)[:30]:  # Limit for speed
                    if m2 == m:
                        continue
                    for _ in range(3):
                        r2 = random.randint(0, m2 - 1)
                        if can_add(temp_set, m2, r2):
                            count += 1
                            break
                if count > best_count:
                    best_count = count
                    best_r = r
            return best_r

    def greedy_construct_with_flexibility():
        """
        Greedy construction with flexibility-based residue selection.
        Process moduli in order of decreasing weight (1/m), choosing residues
        that maximize future packing density.
        """
        best_set = []
        best_score = Fraction(0, 1)
        
        sorted_moduli = sorted(moduli)  # ascending m = descending 1/m
        
        for restart in range(30):
            random.seed(restart * 12345 + 678)
            
            current_set = []
            remaining_moduli = set(sorted_moduli)
            
            for m in sorted_moduli:
                if m not in remaining_moduli:
                    continue
                remaining_moduli.discard(m)
                
                # Choose residue using flexibility scoring
                r = pick_best_residue(current_set, m, remaining_moduli)
                if r is None:
                    # Fallback: try random residues
                    for _ in range(10):
                        r_try = random.randint(0, m - 1)
                        if can_add(current_set, m, r_try):
                            r = r_try
                            break
                if r is not None:
                    current_set.append([m, r])
            
            s = score(current_set)
            if s > best_score:
                best_score = s
                best_set = [x[:] for x in current_set]
                
            if time.time() > deadline - 30:
                break
                
        return best_set

    def local_search(initial_set):
        """
        Improve the set by adding or swapping.
        """
        current_set = [x[:] for x in initial_set]
        current_score = score(current_set)
        
        current_mods = {m for m, r in current_set}
        candidates = [m for m in moduli if m not in current_mods]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try adding a candidate (exhaustive residue search for small m)
            candidates_sorted = sorted(candidates, key=lambda x: 1/x, reverse=True)
            
            for m_new in candidates_sorted:
                found = False
                # Exhaustive search for small moduli, random for large
                if m_new <= 50:
                    for r_new in range(m_new):
                        if can_add(current_set, m_new, r_new):
                            current_set.append([m_new, r_new])
                            current_score += Fraction(1, m_new)
                            current_mods.add(m_new)
                            candidates.remove(m_new)
                            improved = True
                            found = True
                            break
                else:
                    for _ in range(20):
                        r_new = random.randint(0, m_new - 1)
                        if can_add(current_set, m_new, r_new):
                            current_set.append([m_new, r_new])
                            current_score += Fraction(1, m_new)
                            current_mods.add(m_new)
                            candidates.remove(m_new)
                            improved = True
                            found = True
                            break
                
                if found:
                    break
            
            if improved:
                continue
                
            # 2. Try swapping
            if time.time() < deadline - 10:
                for i, (m_old, r_old) in enumerate(current_set):
                    swap_candidates = [m for m in candidates if m < m_old]
                    if not swap_candidates:
                        continue
                    
                    for m_new in swap_candidates:
                        found_swap = False
                        if m_new <= 50:
                            for r_new in range(m_new):
                                temp_set = current_set[:i] + current_set[i+1:]
                                if can_add(temp_set, m_new, r_new):
                                    current_set[i] = [m_new, r_new]
                                    current_score += Fraction(1, m_new) - Fraction(1, m_old)
                                    current_mods.remove(m_old)
                                    current_mods.add(m_new)
                                    candidates.remove(m_new)
                                    candidates.append(m_old)
                                    improved = True
                                    found_swap = True
                                    break
                        else:
                            for _ in range(10):
                                r_new = random.randint(0, m_new - 1)
                                temp_set = current_set[:i] + current_set[i+1:]
                                if can_add(temp_set, m_new, r_new):
                                    current_set[i] = [m_new, r_new]
                                    current_score += Fraction(1, m_new) - Fraction(1, m_old)
                                    current_mods.remove(m_old)
                                    current_mods.add(m_new)
                                    candidates.remove(m_new)
                                    candidates.append(m_old)
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                    if improved:
                        break
            
        return current_set

    def final_polish(current_set):
        """
        Final pass: try to add any remaining moduli with exhaustive residue search.
        Also try removing low-value moduli to make room for higher-value ones.
        """
        current_set = [x[:] for x in current_set]
        current_score = score(current_set)
        current_mods = {m for m, r in current_set}
        candidates = [m for m in moduli if m not in current_mods]
        
        # Try adding all candidates with exhaustive search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            candidates_sorted = sorted(candidates, key=lambda x: 1/x, reverse=True)
            for m_new in candidates_sorted:
                for r_new in range(m_new):
                    if can_add(current_set, m_new, r_new):
                        current_set.append([m_new, r_new])
                        current_score += Fraction(1, m_new)
                        current_mods.add(m_new)
                        candidates.remove(m_new)
                        improved = True
                        break
                if improved:
                    break
        
        # Try swapping: remove a low-value modulus, add a higher-value one
        if time.time() < deadline - 2:
            current_mods = {m for m, r in current_set}
            candidates = [m for m in moduli if m not in current_mods]
            
            for i in range(len(current_set) - 1, -1, -1):  # Remove from end first
                m_old, r_old = current_set[i]
                # Try to replace with any candidate with m_new < m_old
                swap_found = False
                for m_new in sorted(candidates, key=lambda x: 1/x, reverse=True):
                    if m_new >= m_old:
                        break
                    for r_new in range(m_new):
                        temp_set = current_set[:i] + current_set[i+1:]
                        if can_add(temp_set, m_new, r_new):
                            current_set[i] = [m_new, r_new]
                            current_score += Fraction(1, m_new) - Fraction(1, m_old)
                            swap_found = True
                            break
                    if swap_found:
                        break
                if swap_found:
                    pass  # Continue to try more swaps
        
        return current_set

    # Phase 1: Greedy with flexibility
    best_set = greedy_construct_with_flexibility()
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 10:
        refined_set = local_search(best_set)
        if score(refined_set) > score(best_set):
            best_set = refined_set
    
    # Phase 3: Try a few more greedy starts with different seeds
    if time.time() < deadline - 15:
        for extra_restart in range(5):
            if time.time() > deadline - 15:
                break
            random.seed(99999 + extra_restart * 7777)
            current_set = []
            remaining_moduli = set(sorted_moduli)
            sorted_moduli = sorted(moduli)
            for m in sorted_moduli:
                remaining_moduli.discard(m)
                r = pick_best_residue(current_set, m, remaining_moduli)
                if r is None:
                    for _ in range(10):
                        r_try = random.randint(0, m - 1)
                        if can_add(current_set, m, r_try):
                            r = r_try
                            break
                if r is not None:
                    current_set.append([m, r])
            
            refined = local_search(current_set)
            if time.time() < deadline - 3:
                polished = final_polish(refined)
                if score(polished) > score(best_set):
                    best_set = polished
            elif score(refined) > score(best_set):
                best_set = refined
    
    # Phase 4: Final polish on best set
    if time.time() < deadline - 1:
        best_set = final_polish(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
