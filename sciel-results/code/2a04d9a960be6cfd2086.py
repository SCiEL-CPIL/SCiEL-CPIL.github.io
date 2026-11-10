import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find any valid residue for m given selected set. Returns None if none."""
        # Randomize order to avoid bias towards small residues?
        # Or just iterate 0..m-1.
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def calculate_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total

    def get_conflicting_moduli_for_residue(res, m, selected):
        """Return list of indices of moduli in selected that conflict with (m, res)."""
        conflicts = []
        for i, (m2, r2) in enumerate(selected):
            g = gcd(m, m2)
            if res % g == r2 % g:
                conflicts.append(i)
        return conflicts

    def find_blocking_moduli(m, selected):
        """
        For a modulus m that cannot be added, find which existing moduli are blocking it.
        Returns a list of indices of blocking moduli.
        """
        # A modulus m is blocked if for every r in 0..m-1, there is at least one conflict.
        # We want to find moduli that conflict with many potential residues of m.
        
        # Let's count how many residues of m are blocked by each existing modulus.
        # If modulus j blocks residue r, it contributes to the count.
        # If sum of blocks >= m, it might be blocked, but overlaps matter.
        
        # Simpler: For each existing modulus i, check if removing it allows adding m.
        # This is O(N^2 * M). N~100, M~100. 100*100*100 = 1e6. Too slow for every step.
        
        # Heuristic: Pick a random existing modulus, change its residue, see if m can be added.
        # This is what the current code does, but maybe not aggressively enough.
        pass

    # 1. Initial Greedy Construction
    selected = []
    selected_moduli = set()
    
    # Greedy pass: Try to add smallest moduli first
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append([m, r])
            selected_moduli.add(m)
            
    current_score = calculate_score(selected)
    all_moduli = set(range(20, 201))
    missing_moduli = all_moduli - selected_moduli
    
    # 2. Local Search
    no_improvement_count = 0
    max_no_improvement = 20000 # Increased
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Restart with a different random seed or perturbation?
            # For now, just break or continue with more aggressive moves
            pass
            
        improvement_found = False
        
        # Strategy A: Try to add any missing modulus
        # Prioritize smaller moduli (larger reciprocal)
        sorted_missing = sorted(missing_moduli)
        for m in sorted_missing:
            if time.time() > deadline:
                return selected
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
                selected_moduli.add(m)
                missing_moduli.remove(m)
                current_score += Fraction(1, m)
                improvement_found = True
                no_improvement_count = 0
                break # Restart loop to try more adds
        
        if improvement_found:
            continue
            
        # Strategy B: Try to swap
        # Replace a larger modulus with a smaller one to increase score.
        sorted_selected_desc = sorted(selected, key=lambda x: x[0], reverse=True)
        sorted_missing_asc = sorted(missing_moduli)
        
        swap_done = False
        for m_old, r_old in sorted_selected_desc:
            if time.time() > deadline:
                return selected
            # Only consider swaps where m_new < m_old
            for m_new in sorted_missing_asc:
                if m_new >= m_old:
                    continue # 1/m_new <= 1/m_old, no score gain
                
                # Temporarily remove m_old
                idx_old = -1
                for i, (m, r) in enumerate(selected):
                    if m == m_old:
                        idx_old = i
                        break
                if idx_old == -1: continue
                
                selected.pop(idx_old)
                selected_moduli.remove(m_old)
                
                r_new = find_valid_residue(m_new, selected)
                if r_new is not None:
                    # Accept swap
                    selected.append([m_new, r_new])
                    selected_moduli.add(m_new)
                    missing_moduli.remove(m_new)
                    missing_moduli.add(m_old)
                    current_score += Fraction(1, m_new) - Fraction(1, m_old)
                    improvement_found = True
                    swap_done = True
                    no_improvement_count = 0
                    break
                else:
                    # Revert
                    selected.append([m_old, r_old])
                    selected_moduli.add(m_old)
                    
            if swap_done:
                break # Restart loop
        
        if swap_done:
            continue

        # Strategy C: Targeted Residue Perturbation
        # If we are stuck, it means no add is possible and no beneficial swap is possible.
        # This implies that adding any missing modulus would require removing multiple existing ones,
        # or the residues are suboptimal.
        
        # Pick a high-value missing modulus m_target (smallest missing)
        if missing_moduli:
            m_target = min(missing_moduli)
            
            # Try to unblock m_target by changing residues of existing moduli
            # Identify which existing moduli are most likely blocking it.
            # We can try changing a random existing modulus's residue.
            
            if selected:
                # Try a few random perturbations
                for _ in range(50):
                    if time.time() > deadline:
                        return selected
                    idx = random.randint(0, len(selected) - 1)
                    m_cur, r_cur = selected[idx]
                    
                    # Try a new random residue
                    new_r = random.randint(0, m_cur - 1)
                    if new_r == r_cur:
                        continue
                    
                    # Check if this new residue is valid for m_cur
                    # We must ensure m_cur doesn't conflict with others
                    # We can't just change it if it conflicts with others.
                    # So we need to check check_conflict(new_r, m_cur, selected_excluding_idx)
                    
                    # Temporarily remove idx
                    saved = selected.pop(idx)
                    valid = True
                    for m2, r2 in selected:
                        g = gcd(m_cur, m2)
                        if new_r % g == r2 % g:
                            valid = False
                            break
                            
                    if not valid:
                        selected.append(saved)
                        continue
                    
                    # Now try to add m_target
                    r_add = find_valid_residue(m_target, selected)
                    if r_add is not None:
                        # Success!
                        selected.append([m_cur, new_r])
                        selected.append([m_target, r_add])
                        selected_moduli.add(m_target)
                        missing_moduli.remove(m_target)
                        current_score += Fraction(1, m_target)
                        improvement_found = True
                        no_improvement_count = 0
                        break
                    else:
                        # Revert
                        selected.append(saved)

        if not improvement_found:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

    return selected
