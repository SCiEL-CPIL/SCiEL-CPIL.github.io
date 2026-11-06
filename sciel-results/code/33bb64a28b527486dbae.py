"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin
    
    def is_disjoint(m, r, classes):
        for m2, r2 in classes:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def get_score(classes):
        return sum(1.0 / m for m, r in classes)

    # 1. Greedy Initialization
    # Sort moduli by 1/m descending (i.e., m ascending) to prioritize high value
    # But we need to ensure disjointness. 
    # A simple greedy: iterate m from 20 to 200. If we can find a valid r, add it.
    # To maximize score, we want small m.
    # However, a small m might block many others. 
    # Let's try a randomized greedy restart to find a good initial state.
    
    best_initial = None
    best_initial_score = 0
    
    for _ in range(10): # 10 restarts for initialization
        classes = []
        # Shuffle order of moduli to break symmetry, but bias towards smaller?
        # Actually, strict ascending order is good for "packing" small numbers if possible.
        # Let's try ascending m, but randomize residue selection.
        for m in range(20, 201):
            # Try to find a valid residue
            # Randomize the order of residues to find one quickly
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if is_disjoint(m, r, classes):
                    classes.append([m, r])
                    found = True
                    break
        score = get_score(classes)
        if score > best_initial_score:
            best_initial_score = score
            best_initial = [c[:] for c in classes]
            
    # If greedy failed to get anything (unlikely), fallback
    if not best_initial:
        best_initial = [[20, 0]]
        best_initial_score = 1.0/20.0

    current_classes = [c[:] for c in best_initial]
    current_score = best_initial_score
    
    best_classes = [c[:] for c in best_initial]
    best_score = best_initial_score

    # 2. Simulated Annealing
    # Temperature schedule
    T = 0.1  # Initial temperature (scale of score change)
    T_min = 1e-6
    alpha = 0.995 # Cooling rate
    
    # We have ~110s. 
    # Estimate steps: is_disjoint is O(N). N~100. 
    # Let's run as many steps as possible.
    
    steps = 0
    while time.time() < deadline:
        # Move generation
        move_type = random.choice(['add', 'remove', 'swap', 'residue'])
        
        if move_type == 'add':
            # Pick a modulus not in current classes
            used_mods = {m for m, _ in current_classes}
            available_mods = [m for m in range(20, 201) if m not in used_mods]
            if not available_mods:
                continue
            m = random.choice(available_mods)
            # Find a valid residue
            residues = list(range(m))
            random.shuffle(residues)
            valid_r = None
            for r in residues:
                if is_disjoint(m, r, current_classes):
                    valid_r = r
                    break
            if valid_r is not None:
                new_classes = current_classes + [[m, valid_r]]
                new_score = get_score(new_classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
        elif move_type == 'remove':
            if len(current_classes) > 1:
                idx = random.randint(0, len(current_classes) - 1)
                new_classes = [c[:] for i, c in enumerate(current_classes) if i != idx]
                new_score = get_score(new_classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
        elif move_type == 'swap':
            # Swap two moduli? No, moduli are fixed in the set. 
            # "Swap" in this context could mean: Remove one, Add another. 
            # Or: Keep the set of moduli, but change residues? That's 'residue' move.
            # Let's interpret 'swap' as: Pick two moduli, remove one, add the other? 
            # Actually, a more powerful move is: 
            # Pick a modulus m1 in set, pick a modulus m2 not in set.
            # Try to swap m1 for m2.
            # Find valid r for m2 given (current_classes without m1).
            # If valid, apply swap.
            if len(current_classes) == 0:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_old, r_old = current_classes[idx]
            
            used_mods = {m for m, _ in current_classes}
            available_mods = [m for m in range(20, 201) if m not in used_mods]
            if not available_mods:
                continue
            m_new = random.choice(available_mods)
            
            # Check if m_new can fit with the rest of the classes
            rest_classes = [c for i, c in enumerate(current_classes) if i != idx]
            
            residues = list(range(m_new))
            random.shuffle(residues)
            valid_r = None
            for r in residues:
                if is_disjoint(m_new, r, rest_classes):
                    valid_r = r
                    break
            
            if valid_r is not None:
                new_classes = [c[:] for i, c in enumerate(current_classes) if i != idx]
                new_classes.append([m_new, valid_r])
                new_score = get_score(new_classes)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
        elif move_type == 'residue':
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m, r_old = current_classes[idx]
            
            # Find a different valid residue
            # We need r_new such that r_new != r_old and is_disjoint(m, r_new, others)
            # To avoid O(m) scan every time, we can just pick a random r and check.
            # But if m is large, it might take many tries.
            # Better: Pre-compute valid residues? No, depends on others.
            
            # Let's try a few random residues
            found = False
            for _ in range(10):
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                others = [c for i, c in enumerate(current_classes) if i != idx]
                if is_disjoint(m, r_new, others):
                    new_classes = [c[:] for i, c in enumerate(current_classes)]
                    new_classes[idx] = [m, r_new]
                    # Score doesn't change with residue!
                    # So we accept any valid residue change? 
                    # Wait, the score is sum(1/m). Residue doesn't affect score.
                    # So why change residue? 
                    # To enable future moves! 
                    # Changing residue might allow adding a new modulus that was previously blocked.
                    # So we should accept valid residue changes with high probability (or always, since score is neutral).
                    # But we must ensure it's actually different and valid.
                    current_classes = new_classes
                    found = True
                    break
            
            # If we changed a residue, score is same. 
            # We don't update best_score here as it's equal.
            # But we update current_state.

        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
        steps += 1

    return best_classes
# EVOLVE-BLOCK-END
