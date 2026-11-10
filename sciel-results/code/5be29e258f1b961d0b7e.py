"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
from math import gcd

def construct():
    moduli = list(range(20, 201))
    
    # Helper to check if a residue r is valid for modulus m given a set of (m2, r2)
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            if m == m2: continue # Should not happen
            if r % gcd(m, m2) == r2 % gcd(m, m2):
                return False
        return True

    # 1. Initialize with Greedy Solution
    selected = []
    for m in moduli:
        # Try to find any valid r
        # Randomize order of r to break symmetry?
        rs = list(range(m))
        random.shuffle(rs)
        for r in rs:
            if is_valid(m, r, selected):
                selected.append([m, r])
                break

    current_score = sum(1.0 / m for m, r in selected)
    best_score = current_score
    best_selected = [pair[:] for pair in selected]
    
    # 2. Simulated Annealing
    T = 0.5 # Start temperature
    T_min = 0.01
    alpha = 0.995
    max_iter = 100000
    
    # Precompute set of selected moduli for fast lookup in "Add" move
    selected_moduli_set = set(m for m, r in selected)
    
    for i in range(max_iter):
        move_type = random.randint(0, 2)
        delta_score = 0.0
        move_executed = False
        
        if move_type == 0:
            # Add
            unselected = [m for m in moduli if m not in selected_moduli_set]
            if unselected:
                m = random.choice(unselected)
                r = random.randint(0, m - 1)
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    selected_moduli_set.add(m)
                    delta_score = 1.0 / m
                    move_executed = True
                    
        elif move_type == 1:
            # Remove
            if selected:
                idx = random.randint(0, len(selected) - 1)
                m, r = selected[idx]
                selected.pop(idx)
                selected_moduli_set.remove(m)
                delta_score = -1.0 / m
                move_executed = True
                
        else:
            # Change Residue
            if selected:
                idx = random.randint(0, len(selected) - 1)
                m, old_r = selected[idx]
                # Pick a different r
                r = random.randint(0, m - 1)
                while r == old_r:
                    r = random.randint(0, m - 1)
                if is_valid(m, r, selected):
                    selected[idx] = [m, r]
                    delta_score = 0.0
                    move_executed = True

        if not move_executed:
            continue

        new_score = current_score + delta_score
        
        # Acceptance
        if new_score > current_score or random.random() < math.exp(delta_score / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_selected = [pair[:] for pair in selected]
        else:
            # Revert move
            if move_type == 0:
                selected.pop()
                selected_moduli_set.remove(m)
            elif move_type == 1:
                selected.append([m, r])
                selected_moduli_set.add(m)
            else:
                selected[idx] = [m, old_r]

        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    return best_selected

# EVOLVE-BLOCK-END
