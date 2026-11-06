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

    # 1. Greedy Initialization: All Evens
    # Start with all even moduli in [20, 200]
    initial_classes = []
    # We need to assign residues.
    # Greedy: for each m, find a valid r.
    # If no valid r, skip m (but we hope there is one).
    
    best_initial = None
    best_initial_score = 0
    
    # Try a few times with random residue ordering to find a valid assignment for all evens
    for _ in range(50):
        classes = []
        success = True
        for m in range(20, 201, 2): # Evens only
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if is_disjoint(m, r, classes):
                    classes.append([m, r])
                    found = True
                    break
            if not found:
                success = False
                break
        
        if success:
            score = get_score(classes)
            if score > best_initial_score:
                best_initial_score = score
                best_initial = [c[:] for c in classes]
                
    if not best_initial:
        # Fallback: just 20
        best_initial = [[20, 0]]
        best_initial_score = 1.0/20.0

    current_classes = [c[:] for c in best_initial]
    current_score = best_initial_score
    
    best_classes = [c[:] for c in best_initial]
    best_score = best_initial_score

    # 2. Simulated Annealing
    T = 0.05
    T_min = 1e-6
    alpha = 0.995
    
    steps = 0
    while time.time() < deadline:
        # Move generation
        # Focus on adding missing moduli (odds or skipped evens) and swapping residues
        move_type = random.choice(['add', 'remove', 'swap', 'residue'])
        
        if move_type == 'add':
            used_mods = {m for m, _ in current_classes}
            available_mods = [m for m in range(20, 201) if m not in used_mods]
            if not available_mods:
                continue
            m = random.choice(available_mods)
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
            if len(current_classes) == 0:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_old, r_old = current_classes[idx]
            
            used_mods = {m for m, _ in current_classes}
            available_mods = [m for m in range(20, 201) if m not in used_mods]
            if not available_mods:
                continue
            m_new = random.choice(available_mods)
            
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
            
            found = False
            for _ in range(10):
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                others = [c for i, c in enumerate(current_classes) if i != idx]
                if is_disjoint(m, r_new, others):
                    new_classes = [c[:] for i, c in enumerate(current_classes)]
                    new_classes[idx] = [m, r_new]
                    current_classes = new_classes
                    found = True
                    break

        T *= alpha
        if T < T_min:
            T = T_min
        steps += 1

    return best_classes
# EVOLVE-BLOCK-END
