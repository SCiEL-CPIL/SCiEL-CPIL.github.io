import math
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    def is_valid_assignment(selected, m, r):
        """Check if adding (m, r) to selected keeps all pairs disjoint."""
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_residue(selected, m):
        """Find a valid residue for modulus m given current selected set."""
        # Try random residues first to avoid bias, then fallback to sequential
        # But for speed, sequential is fine if we randomize the order of moduli in the main loop
        for r in range(m):
            if is_valid_assignment(selected, m, r):
                return r
        return None

    def compute_score(selected):
        return sum(1.0 / m for m, r in selected)

    # Phase 1: Greedy construction with multiple random orderings
    best_selected = []
    best_score = 0.0
    
    # We will run several greedy passes with different random orderings of moduli
    # and random residue selection (if multiple valid residues exist, pick one randomly)
    
    def greedy_construct(order):
        selected = []
        for m in order:
            # Find ALL valid residues, pick one randomly
            valid_residues = []
            for r in range(m):
                if is_valid_assignment(selected, m, r):
                    valid_residues.append(r)
            if valid_residues:
                r = random.choice(valid_residues)
                selected.append((m, r))
        return selected

    # Initial greedy with sorted order (small to large)
    moduli_sorted = list(range(20, 201))
    selected = greedy_construct(moduli_sorted)
    best_score = compute_score(selected)
    best_selected = list(selected)
    
    # Random restarts
    random.seed(42)
    for i in range(100): # 100 restarts
        if time.time() > deadline - 20:
            break
        random_order = list(range(20, 201))
        random.shuffle(random_order)
        selected = greedy_construct(random_order)
        score = compute_score(selected)
        if score > best_score:
            best_score = score
            best_selected = list(selected)
            
    # Phase 2: Local Search on the best solution found
    # Moves: 
    # 1. Add a missing modulus
    # 2. Remove a modulus
    # 3. Swap a modulus
    # 4. Change residue of a modulus (to enable additions)
    
    current = list(best_selected)
    current_score = best_score
    
    def try_add(current, current_score):
        selected_moduli = {m for m, r in current}
        for m in range(20, 201):
            if m in selected_moduli:
                continue
            r = find_valid_residue(current, m)
            if r is not None:
                current.append((m, r))
                return current, current_score + 1.0 / m
        return current, current_score

    def try_swap(current, current_score):
        # Try to replace one modulus with another
        # This is expensive, so we limit it
        for i in range(len(current)):
            m_old, r_old = current[i]
            # Try replacing with a smaller modulus not in set
            selected_moduli = {m for m, r in current}
            for m_new in range(20, 201):
                if m_new in selected_moduli:
                    continue
                if m_new >= m_old:
                    break # We want smaller moduli for higher score, but must be valid
                # Try to find a residue for m_new
                # We need to check if m_new can coexist with all others in current except m_old
                temp_current = [x for j, x in enumerate(current) if j != i]
                r_new = find_valid_residue(temp_current, m_new)
                if r_new is not None:
                    # Check if the score improves
                    new_score = current_score - 1.0 / m_old + 1.0 / m_new
                    if new_score > current_score + 1e-12:
                        current[i] = (m_new, r_new)
                        return current, new_score
        return current, current_score

    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        no_improve_count = 0
        
        # Try adding
        new_current, new_score = try_add(current, current_score)
        if new_score > current_score + 1e-12:
            current = new_current
            current_score = new_score
            improved = True
            continue
            
        # Try swapping
        new_current, new_score = try_swap(current, current_score)
        if new_score > current_score + 1e-12:
            current = new_current
            current_score = new_score
            improved = True
            continue
            
        # If no improvement, try random perturbation (change a residue)
        if no_improve_count > 10:
            # Pick a random modulus and change its residue
            if current:
                idx = random.randint(0, len(current) - 1)
                m, r = current[idx]
                # Try random residues
                for _ in range(10):
                    r_new = random.randint(0, m - 1)
                    # Check validity
                    valid = True
                    for j, (m_j, r_j) in enumerate(current):
                        if i == j: continue
                        g = math.gcd(m, m_j)
                        if r_new % g == r_j % g:
                            valid = False
                            break
                    if valid:
                        current[idx] = (m, r_new)
                        # Try to add after perturbation
                        new_current, new_score = try_add(current, current_score)
                        if new_score > current_score + 1e-12:
                            current = new_current
                            current_score = new_score
                            improved = True
                            break
            no_improve_count += 1
            if no_improve_count > 50:
                break

    if current_score > best_score:
        best_selected = current
        best_score = current_score

    return [[m, r] for m, r in best_selected]
