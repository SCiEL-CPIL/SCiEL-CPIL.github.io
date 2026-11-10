import time
import random
from math import gcd
from fractions import Fraction

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
        """Find any valid residue for m given selected set. Returns None if none."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def calculate_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total

    # 1. Initial Greedy Construction
    selected = []
    selected_moduli_set = set()
    
    # Sort moduli by 1/m descending (m ascending)
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append([m, r])
            selected_moduli_set.add(m)

    current_score = calculate_score(selected)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli_set]
    
    # 2. Local Search
    no_improvement_count = 0
    max_no_improvement = 2000
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        
        # Strategy A: Try to add a missing modulus directly
        if missing_moduli:
            # Bias towards smaller m
            target_m = missing_moduli[0]
            if random.random() < 0.1 and len(missing_moduli) > 1:
                target_m = random.choice(missing_moduli)
                
            target_r = find_valid_residue(target_m, selected)
            if target_r is not None:
                selected.append([target_m, target_r])
                selected_moduli_set.add(target_m)
                missing_moduli.remove(target_m)
                current_score += Fraction(1, target_m)
                improved = True
                no_improvement_count = 0
                continue
        
        # Strategy B: Swap
        if not missing_moduli:
            break
            
        target_m = missing_moduli[0]
        
        # Find victims larger than target_m
        # We want to swap large m for small m
        victims = []
        for i, (m, r) in enumerate(selected):
            if m > target_m:
                victims.append((m, i))
        
        if victims:
            # Sort by m descending to try swapping the largest ones first
            victims.sort(key=lambda x: -x[0])
            
            # Try top 5 victims
            for victim_m, v_idx in victims[:5]:
                if time.time() > deadline:
                    break
                
                victim_r = selected[v_idx][1]
                
                # Remove victim
                selected.pop(v_idx)
                
                # Try to add target
                target_r = find_valid_residue(target_m, selected)
                if target_r is not None:
                    # Add target
                    selected.append([target_m, target_r])
                    
                    # Try to re-add victim
                    victim_r_new = find_valid_residue(victim_m, selected)
                    if victim_r_new is not None:
                        # Re-add victim
                        selected.append([victim_m, victim_r_new])
                        # Score increased by 1/target_m
                        current_score += Fraction(1, target_m)
                        missing_moduli.remove(target_m)
                        improved = True
                    else:
                        # Swap
                        current_score += Fraction(1, target_m) - Fraction(1, victim_m)
                        missing_moduli.remove(target_m)
                        missing_moduli.append(victim_m)
                        missing_moduli.sort()
                        improved = True
                    break
                else:
                    # Revert
                    selected.append([victim_m, victim_r])
        
        # Strategy C: Perturb residue
        if not improved and selected:
            p_idx = random.randint(0, len(selected) - 1)
            p_m, p_r = selected[p_idx]
            old_r = p_r
            # Try a few random new residues
            for _ in range(10):
                new_r = random.randint(0, p_m - 1)
                if new_r == old_r:
                    continue
                selected[p_idx][1] = new_r
                
                # Check if smallest missing can be added
                if missing_moduli:
                    t_m = missing_moduli[0]
                    t_r = find_valid_residue(t_m, selected)
                    if t_r is not None:
                        selected.append([t_m, t_r])
                        missing_moduli.remove(t_m)
                        current_score += Fraction(1, t_m)
                        improved = True
                        break
                else:
                    break
                    
            if not improved:
                selected[p_idx][1] = old_r

        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    return selected
