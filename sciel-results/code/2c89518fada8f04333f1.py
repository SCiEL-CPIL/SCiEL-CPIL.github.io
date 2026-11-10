"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # Precompute gcd table for all pairs of moduli in [20, 200]
    moduli_range = list(range(20, 201))
    gcd_table = {}
    for i, m1 in enumerate(moduli_range):
        for j, m2 in enumerate(moduli_range):
            if i != j:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
    def check_conflict_fast(res, m, selected_moduli, selected_residues):
        """Check if residue res with modulus m conflicts with any selected class."""
        for idx in range(len(selected_moduli)):
            m2 = selected_moduli[idx]
            r2 = selected_residues[idx]
            g = gcd_table.get((m, m2), gcd(m, m2))
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue_fast(m, selected_moduli, selected_residues):
        """Find a valid residue for modulus m. Returns None if impossible."""
        for r in range(m):
            if not check_conflict_fast(r, m, selected_moduli, selected_residues):
                return r
        return None

    def score_from_list(selected):
        """Compute score from list of [m, r] pairs."""
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return float(total)

    def greedy_fill_ordered(selected, order):
        """Fill with moduli in the given order."""
        existing_moduli = {m for m, r in selected}
        sel_mods = [m for m, r in selected]
        sel_res = [r for m, r in selected]
        for m in order:
            if m in existing_moduli:
                continue
            r = find_valid_residue_fast(m, sel_mods, sel_res)
            if r is not None:
                selected.append([m, r])
                sel_mods.append(m)
                sel_res.append(r)
                existing_moduli.add(m)
        return selected

    def try_add_and_greedy(selected, m_add, fill_order):
        """Try adding m_add then greedy fill with remaining moduli."""
        if m_add in {m for m, r in selected}:
            return None
        temp_selected = [x[:] for x in selected]
        sel_mods = [m for m, r in temp_selected]
        sel_res = [r for m, r in temp_selected]
        r = find_valid_residue_fast(m_add, sel_mods, sel_res)
        if r is None:
            return None
        temp_selected.append([m_add, r])
        sel_mods.append(m_add)
        sel_res.append(r)
        existing_moduli = {m for m, r in temp_selected}
        for m in fill_order:
            if m in existing_moduli:
                continue
            r2 = find_valid_residue_fast(m, sel_mods, sel_res)
            if r2 is not None:
                temp_selected.append([m, r2])
                sel_mods.append(m)
                sel_res.append(r2)
                existing_moduli.add(m)
        return temp_selected

    # Phase 1: Multiple greedy starts with different orderings
    best_selected = []
    best_score = 0.0
    
    # Standard increasing order
    order_increasing = list(range(20, 201))
    selected = greedy_fill_ordered([], order_increasing)
    best_selected = [x[:] for x in selected]
    best_score = score_from_list(selected)
    
    # Try a few different orderings to escape local optima
    random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 30:
            break
        # Create a perturbed order
        order = list(range(20, 201))
        # Shuffle the upper half
        upper_half = order[60:]
        random.shuffle(upper_half)
        order = order[:60] + upper_half
        
        selected_trial = greedy_fill_ordered([], order)
        trial_score = score_from_list(selected_trial)
        if trial_score > best_score:
            best_score = trial_score
            best_selected = [x[:] for x in selected_trial]
    
    if time.time() > deadline - 10:
        return best_selected
    
    # Phase 2: Iterative improvement
    current_selected = [x[:] for x in best_selected]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        
        # Step A: Try to add any missing modulus
        existing_moduli = {m for m, r in current_selected}
        candidates_to_add = [m for m in range(20, 201) if m not in existing_moduli]
        
        # Try adding in order of 1/m (smallest first)
        for m_in in candidates_to_add:
            temp = try_add_and_greedy(current_selected, m_in, list(range(20, 201)))
            if temp is not None:
                new_score = score_from_list(temp)
                if new_score > current_score:
                    current_selected = temp
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [x[:] for x in current_selected]
                    improved = True
                    no_improve_count = 0
                    break
        
        if improved:
            continue
        
        # Step B: Try single swaps
        selected_sorted_desc = sorted(current_selected, key=lambda x: x[0], reverse=True)
        candidates_to_swap_in = [m for m in range(20, 201) if m not in existing_moduli]
        
        swap_performed = False
        for m_out_val, r_out in selected_sorted_desc:
            for m_in in candidates_to_swap_in:
                if m_in >= m_out_val:
                    break
                
                temp_selected = [x[:] for x in current_selected if x[0] != m_out_val]
                sel_mods = [m for m, r in temp_selected]
                sel_res = [r for m, r in temp_selected]
                r = find_valid_residue_fast(m_in, sel_mods, sel_res)
                if r is not None:
                    temp_selected.append([m_in, r])
                    sel_mods.append(m_in)
                    sel_res.append(r)
                    # Greedy fill remaining
                    existing = {m for m, r in temp_selected}
                    for m in range(20, 201):
                        if m in existing:
                            continue
                        r2 = find_valid_residue_fast(m, sel_mods, sel_res)
                        if r2 is not None:
                            temp_selected.append([m, r2])
                            sel_mods.append(m)
                            sel_res.append(r2)
                            existing.add(m)
                    
                    new_score = score_from_list(temp_selected)
                    if new_score > current_score:
                        current_selected = temp_selected
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_selected = [x[:] for x in current_selected]
                        swap_performed = True
                        no_improve_count = 0
                        break
            
            if swap_performed:
                break
        
        if swap_performed:
            continue
        
        # Step C: Try double swaps (remove 2 large, add 2 small)
        if len(current_selected) >= 2:
            existing_moduli = {m for m, r in current_selected}
            candidates_small = [m for m in range(20, 201) if m not in existing_moduli]
            
            double_swap_performed = False
            if len(candidates_small) >= 2:
                # Try top 8 small candidates
                small_candidates = candidates_small[:8]
                # Try top 6 large to remove
                large_to_remove = [x[0] for x in selected_sorted_desc[:6]]
                
                for i in range(len(small_candidates)):
                    if double_swap_performed:
                        break
                    for j in range(i + 1, len(small_candidates)):
                        if double_swap_performed:
                            break
                        m_in1, m_in2 = small_candidates[i], small_candidates[j]
                        
                        for k in range(len(large_to_remove)):
                            if double_swap_performed:
                                break
                            for l in range(k + 1, len(large_to_remove)):
                                m_out1, m_out2 = large_to_remove[k], large_to_remove[l]
                                
                                if m_in1 >= m_out1 or m_in2 >= m_out2:
                                    continue
                                
                                temp_selected = [x[:] for x in current_selected
                                                 if x[0] != m_out1 and x[0] != m_out2]
                                sel_mods = [m for m, r in temp_selected]
                                sel_res = [r for m, r in temp_selected]
                                
                                r1 = find_valid_residue_fast(m_in1, sel_mods, sel_res)
                                if r1 is None:
                                    continue
                                temp_selected.append([m_in1, r1])
                                sel_mods.append(m_in1)
                                sel_res.append(r1)
                                
                                r2 = find_valid_residue_fast(m_in2, sel_mods, sel_res)
                                if r2 is None:
                                    continue
                                temp_selected.append([m_in2, r2])
                                sel_mods.append(m_in2)
                                sel_res.append(r2)
                                
                                # Greedy fill
                                existing = {m for m, r in temp_selected}
                                for m in range(20, 201):
                                    if m in existing:
                                        continue
                                    r3 = find_valid_residue_fast(m, sel_mods, sel_res)
                                    if r3 is not None:
                                        temp_selected.append([m, r3])
                                        sel_mods.append(m)
                                        sel_res.append(r3)
                                        existing.add(m)
                                
                                new_score = score_from_list(temp_selected)
                                if new_score > current_score:
                                    current_selected = temp_selected
                                    current_score = new_score
                                    if new_score > best_score:
                                        best_score = new_score
                                        best_selected = [x[:] for x in current_selected]
                                    double_swap_performed = True
                                    no_improve_count = 0
                                    break
        
        if double_swap_performed:
            continue
        
        # Step D: Perturbation - remove random moduli and re-greedy
        if len(current_selected) > 10:
            n_remove = random.randint(2, min(4, len(current_selected) - 2))
            sel_sorted_by_size = sorted(range(len(current_selected)), key=lambda i: -current_selected[i][0])
            # Remove from the larger moduli
            top_half = sel_sorted_by_size[:max(2, len(sel_sorted_by_size) // 2)]
            to_remove = set(random.sample(top_half, min(n_remove, len(top_half))))
            
            temp_selected = []
            for i, item in enumerate(current_selected):
                if i not in to_remove:
                    temp_selected.append(item[:])
            
            # Re-greedy with standard order
            temp_selected = greedy_fill_ordered(temp_selected, list(range(20, 201)))
            new_score = score_from_list(temp_selected)
            
            if new_score > current_score:
                current_selected = temp_selected
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in current_selected]
                no_improve_count = 0
                continue
        
        # No improvement this iteration
        no_improve_count += 1
        if no_improve_count >= max_no_improve:
            # Reset to best and try a fresh perturbation
            current_selected = [x[:] for x in best_selected]
            current_score = best_score
            no_improve_count = 0
            
            # Aggressive perturbation
            if len(current_selected) > 5:
                n_remove = random.randint(3, min(5, len(current_selected) - 3))
                indices = list(range(len(current_selected)))
                random.shuffle(indices)
                to_remove = set(indices[:n_remove])
                
                temp_selected = []
                for i, item in enumerate(current_selected):
                    if i not in to_remove:
                        temp_selected.append(item[:])
                
                temp_selected = greedy_fill_ordered(temp_selected, list(range(20, 201)))
                new_score = score_from_list(temp_selected)
                if new_score > current_score:
                    current_selected = temp_selected
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [x[:] for x in current_selected]

    return best_selected
# EVOLVE-BLOCK-END
