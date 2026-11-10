import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

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

    def is_valid_set(selected):
        """Check if the entire set is valid."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    # 1. Initial Greedy Construction
    # Try multiple orders to find a good starting point
    best_selected = []
    best_score = Fraction(0, 1)
    
    # Order 1: Ascending m (standard greedy)
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, best_selected)
        if r is not None:
            best_selected.append([m, r])
    score1 = calculate_score(best_selected)
    
    # Order 2: Descending m (try large moduli first to see if it helps, unlikely but cheap)
    # Actually, for maximizing sum 1/m, we want small m. So ascending is best.
    # Let's try a different strategy: Randomized Greedy with restarts
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        selected = []
        # Shuffle the order of moduli slightly, but keep small ones prioritized
        # Actually, let's just try adding them in ascending order but with random tie-breaking?
        # No, the conflict depends on residues.
        # Let's try a "randomized greedy" where we pick the next modulus that fits, 
        # preferring smaller m, but with some randomness in residue choice.
        
        # Actually, the greedy that picks smallest m first is usually optimal for this type of packing.
        # Let's stick to the deterministic greedy as the base, but maybe try a few variations of residue selection.
        
        # Variation: When adding m, instead of picking the first valid residue, pick a random valid one?
        # This might lead to different local optima.
        
        selected = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            # Find all valid residues
            valid_res = []
            for r in range(m):
                if not check_conflict(r, m, selected):
                    valid_res.append(r)
            if valid_res:
                # Pick a random one to explore different paths
                r = random.choice(valid_res)
                selected.append([m, r])
        
        score = calculate_score(selected)
        if score > best_score:
            best_score = score
            best_selected = [x[:] for x in selected]
            
    selected = best_selected
    current_score = best_score
    
    # Identify missing moduli
    selected_moduli = set(m for m, r in selected)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
    missing_moduli.sort()

    # 2. Local Search
    # We use a more robust local search:
    # - Try to add a missing modulus by perturbing existing residues.
    # - Try to swap a selected modulus with a missing one if it improves score.
    # - Try to remove a selected modulus to allow adding two missing ones (if possible).
    
    iterations = 0
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline:
        iterations += 1
        
        if not missing_moduli:
            break
            
        # Strategy 1: Add a missing modulus by perturbing one existing residue
        # Pick a target missing modulus (prefer smaller m for higher 1/m)
        # But sometimes a larger m is easier to add. Let's try a mix.
        if random.random() < 0.7:
            target_m = missing_moduli[0]
        else:
            target_m = random.choice(missing_moduli)
            
        if not selected:
            break
            
        # Pick a random existing modulus to perturb
        victim_idx = random.randint(0, len(selected) - 1)
        victim_m, victim_r = selected[victim_idx]
        
        found_improvement = False
        
        # Try a few random new residues for victim
        # Limit attempts to keep it fast
        attempts = 20
        for _ in range(attempts):
            new_victim_r = random.randint(0, victim_m - 1)
            if new_victim_r == victim_r:
                continue
                
            old_r = selected[victim_idx][1]
            selected[victim_idx][1] = new_victim_r
            
            # Check if target_m can be added now
            target_r = find_valid_residue(target_m, selected)
            if target_r is not None:
                # We can add target_m!
                new_score = current_score + Fraction(1, target_m)
                # Since we're adding a modulus, score always increases
                selected.append([target_m, target_r])
                current_score = new_score
                if target_m in missing_moduli:
                    missing_moduli.remove(target_m)
                found_improvement = True
                no_improvement_count = 0
                break
            
            # Revert victim if not accepted
            selected[victim_idx][1] = old_r
            
        if found_improvement:
            continue
            
        # Strategy 2: Swap a selected modulus for a missing one
        # Pick a random selected modulus (victim) and a random missing modulus (target)
        # If target_m < victim_m, then 1/target_m > 1/victim_m, so swap is beneficial if valid.
        
        # Pick victim
        victim_idx = random.randint(0, len(selected) - 1)
        victim_m, victim_r = selected[victim_idx]
        
        # Pick target
        # Prefer targets with smaller m (higher value)
        # Filter missing moduli that are smaller than victim_m
        better_targets = [m for m in missing_moduli if m < victim_m]
        if better_targets:
            target_m = random.choice(better_targets)
        else:
            # If no better targets, maybe try a random one just to explore?
            # But score would decrease. Only do this if we are stuck?
            # For now, skip if no better targets.
            continue
            
        # Remove victim
        selected.pop(victim_idx)
        
        # Try to add target
        target_r = find_valid_residue(target_m, selected)
        if target_r is not None:
            # Swap successful and valid
            selected.append([target_m, target_r])
            current_score += Fraction(1, target_m) - Fraction(1, victim_m)
            if target_m in missing_moduli:
                missing_moduli.remove(target_m)
            missing_moduli.append(victim_m)
            missing_moduli.sort()
            no_improvement_count = 0
        else:
            # Revert
            selected.append([victim_m, victim_r])
            
        # Strategy 3: Remove a selected modulus to see if it allows adding multiple missing ones
        # This is complex. Let's stick to simpler moves for now.
        
        # Track no improvement
        # We didn't find an improvement in this iteration
        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            # Random restart of local search? Or just break?
            # Let's try a different random seed or break.
            # Breaking might be too early. Let's just continue but maybe change strategy.
            # For now, just continue.
            pass

    # Final validation
    if not is_valid_set(selected):
        # Fallback to greedy if something went wrong
        selected = []
        for m in range(20, 201):
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])

    return selected
