"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Moduli range
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_set = set(moduli)
    
    # Precompute gcd table for speed
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for m2 in moduli[i:]:
            g = gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            if m1 != m2:
                gcd_table[(m2, m1)] = g

    def is_valid(classes, m, r):
        """Check if adding (m, r) to classes is valid."""
        for (m2, r2) in classes:
            if m2 == m:
                return False
            g = gcd_table.get((m, m2), 1)
            if (r - r2) % g == 0:
                return False
        return True

    def find_valid_residue(classes, m):
        """Find any valid residue for m given classes. Returns None if not possible."""
        # Randomize order of residues to escape symmetry? 
        # Or just pick the first one.
        # For SA, we might want to pick a random valid one to explore.
        valid_residues = []
        for r in range(m):
            if is_valid(classes, m, r):
                valid_residues.append(r)
        if not valid_residues:
            return None
        return random.choice(valid_residues)

    def score(classes):
        return sum(1.0 / m for m, r in classes)

    def greedy_construct():
        """Greedy construction: add classes in order of increasing modulus."""
        classes = []
        used_moduli = set()
        for m in sorted(moduli):
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    def local_polish(classes):
        """Greedy add of any missing moduli that fit."""
        used_moduli = set(m for m, r in classes)
        improved = True
        while improved:
            improved = False
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r = find_valid_residue(classes, m)
                if r is not None:
                    classes.append([m, r])
                    used_moduli.add(m)
                    improved = True
        return classes

    # 1. Initial Greedy Solution
    best_classes = greedy_construct()
    best_classes = local_polish(best_classes)
    best_score = score(best_classes)
    current_classes = [list(c) for c in best_classes]
    current_score = best_score

    # 2. Simulated Annealing
    # State: list of [m, r]
    # Moves: Add, Remove, Change Residue
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    max_steps = 200000 # Upper bound, time controlled
    
    # Precompute list of moduli for random selection
    all_moduli = moduli
    
    while time.time() < deadline - 10:
        if T < min_T:
            break
            
        # Generate a move
        move_type = random.random()
        
        new_classes = [list(c) for c in current_classes]
        new_score = current_score
        valid_move = False
        
        if move_type < 0.4:
            # Add a random modulus
            # Prefer smaller moduli? 
            # Let's pick randomly from all, but weight towards smaller?
            # Simple random from all_moduli
            m_new = random.choice(all_moduli)
            if any(c[0] == m_new for c in current_classes):
                continue # Skip if already present
            
            r_new = find_valid_residue(current_classes, m_new)
            if r_new is not None:
                new_classes.append([m_new, r_new])
                new_score = current_score + 1.0 / m_new
                valid_move = True
                
        elif move_type < 0.7:
            # Remove a random modulus
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_rem, r_rem = current_classes[idx]
            new_classes.pop(idx)
            new_score = current_score - 1.0 / m_rem
            valid_move = True
            
        else:
            # Change residue of a random modulus
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_chg, r_chg = current_classes[idx]
            
            # Find a different valid residue
            # To avoid checking all m, just pick a random r and check?
            # Or find all valid residues and pick a different one.
            # For speed, let's try a few random residues
            found_new_r = False
            for _ in range(5): # Try 5 random residues
                r_try = random.randint(0, m_chg - 1)
                if r_try == r_chg:
                    continue
                # Check validity against others
                others = [c for i, c in enumerate(current_classes) if i != idx]
                if is_valid(others, m_chg, r_try):
                    new_classes[idx] = [m_chg, r_try]
                    # Score doesn't change
                    valid_move = True
                    found_new_r = True
                    break
            
            if not found_new_r:
                continue
        
        if not valid_move:
            continue
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Always accept better
            current_classes = new_classes
            current_score = new_score
        else:
            # Accept worse with probability exp(delta / T)
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_classes = new_classes
                current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_classes = [list(c) for c in current_classes]
            
        # Periodic local polish to fill gaps
        if random.random() < 0.01: # 1% chance
            polished = local_polish(current_classes)
            p_score = score(polished)
            if p_score >= current_score:
                current_classes = polished
                current_score = p_score
                if current_score > best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in current_classes]
            
        # Cool down
        T *= cooling_rate

    # Final polish
    best_classes = local_polish(best_classes)
    best_score = score(best_classes)
    
    return best_classes
# EVOLVE-BLOCK-END
