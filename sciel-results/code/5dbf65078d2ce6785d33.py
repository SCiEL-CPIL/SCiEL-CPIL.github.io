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
        # Try 0..m-1
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
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append([m, r])

    current_score = calculate_score(selected)
    
    # Identify missing moduli
    selected_moduli = set(m for m, r in selected)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
    
    # Sort missing moduli by value (1/m) descending, i.e., m ascending
    missing_moduli.sort()

    # 2. Local Search
    # We try to add missing moduli by modifying existing ones or swapping.
    
    # Helper to get index of modulus in selected
    def get_index(m):
        for i, (m_i, r_i) in enumerate(selected):
            if m_i == m:
                return i
        return -1

    # Precompute gcds? No, N is small.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations % 1000 == 0:
            # Optional: print or debug, but not needed.
            pass

        if not missing_moduli:
            break
            
        # Strategy A: Try to add a missing modulus by changing one existing residue
        # Pick a target missing modulus (prefer smaller m)
        target_m = missing_moduli[0] # Smallest missing
        
        # Pick a random existing modulus to perturb
        if not selected:
            break
        victim_idx = random.randint(0, len(selected) - 1)
        victim_m, victim_r = selected[victim_idx]
        
        # Try random new residue for victim
        # To be efficient, try a few random residues
        found_improvement = False
        for _ in range(10): # Try up to 10 random perturbations
            new_victim_r = random.randint(0, victim_m - 1)
            if new_victim_r == victim_r:
                continue
                
            # Temporarily update victim
            old_r = selected[victim_idx][1]
            selected[victim_idx][1] = new_victim_r
            
            # Check if target_m can be added now
            target_r = find_valid_residue(target_m, selected)
            if target_r is not None:
                # We can add target_m!
                # Calculate new score
                new_score = current_score + Fraction(1, target_m)
                if new_score > current_score:
                    # Accept
                    selected.append([target_m, target_r])
                    current_score = new_score
                    # Update missing list
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                    found_improvement = True
                    break
            # Revert victim if not accepted
            selected[victim_idx][1] = old_r
            
        if found_improvement:
            continue

        # Strategy B: Try to swap a victim for a target
        # This is harder. Let's try a different approach:
        # Randomly pick a victim, remove it, then try to add target.
        # If target can be added, try to re-add victim.
        
        # Pick random target from missing (not just the smallest, to explore)
        if random.random() < 0.5:
             target_m = missing_moduli[0]
        else:
             target_m = random.choice(missing_moduli)
             
        if not selected:
             break
             
        victim_idx = random.randint(0, len(selected) - 1)
        victim_m, victim_r = selected[victim_idx]
        
        # Remove victim
        selected.pop(victim_idx)
        
        # Try to add target
        target_r = find_valid_residue(target_m, selected)
        if target_r is not None:
            # Check if victim can be re-added
            victim_r_new = find_valid_residue(victim_m, selected)
            if victim_r_new is not None:
                # We have a valid set with target instead of victim?
                # Wait, if we re-add victim, we have both.
                # Score change: +1/target_m - 1/victim_m
                # Since target_m is missing, it wasn't in the set.
                # If we can have BOTH, that's an improvement.
                # But we removed victim to make space? No, removing victim doesn't "make space" in terms of count, it just changes constraints.
                # If we can add target AND re-add victim, then we just added target.
                # This is effectively the same as Strategy A but with a full removal/re-add.
                
                # Let's just add target to the current set (which has victim removed)
                selected.append([target_m, target_r])
                # Now try to re-add victim
                victim_r_new = find_valid_residue(victim_m, selected)
                if victim_r_new is not None:
                    selected.append([victim_m, victim_r_new])
                    # Score increased by 1/target_m
                    current_score += Fraction(1, target_m)
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                else:
                    # Victim cannot be re-added. We swapped victim for target.
                    # Score change: +1/target_m - 1/victim_m
                    score_change = Fraction(1, target_m) - Fraction(1, victim_m)
                    if score_change > 0:
                        current_score += score_change
                        if target_m in missing_moduli:
                            missing_moduli.remove(target_m)
                        # victim_m is now missing
                        missing_moduli.append(victim_m)
                        missing_moduli.sort()
                    else:
                        # Revert: Remove target, re-add victim
                        selected.pop() # Remove target
                        selected.append([victim_m, victim_r])
            else:
                # Victim cannot be re-added, and we don't want to lose it if target is not better
                # Actually, if victim cannot be re-added, we have a set with target but not victim.
                # Score change: +1/target_m - 1/victim_m
                score_change = Fraction(1, target_m) - Fraction(1, victim_m)
                if score_change > 0:
                    current_score += score_change
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                    missing_moduli.append(victim_m)
                    missing_moduli.sort()
                else:
                    # Revert: Remove target, re-add victim
                    selected.pop() # Remove target
                    selected.append([victim_m, victim_r])
        else:
            # Target cannot be added even without victim. Revert removal of victim.
            selected.append([victim_m, victim_r])

    return selected
