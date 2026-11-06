import time
from math import gcd
import random

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def _is_valid_addition(m, r, selected):
    """Check if adding (m, r) is valid given the selected set."""
    for m2, r2 in selected.items():
        g = gcd(m, m2)
        if (r - r2) % g == 0:
            return False
    return True

def _greedy_fill(initial_classes, deadline):
    """
    Fills the set starting from initial_classes by adding more moduli
    in ascending order of m (highest 1/m first).
    """
    selected = {m: r for m, r in initial_classes}
    
    all_moduli = list(range(20, 201))
    candidates = [m for m in all_moduli if m not in selected]
    candidates.sort(key=lambda m: m)
    
    for m in candidates:
        if time.time() > deadline:
            break
            
        found = False
        for r in range(m):
            if _is_valid_addition(m, r, selected):
                found = True
                selected[m] = r
                break
        
    return [[m, r] for m, r in selected.items()]

def _simulate_annealing(classes, deadline, initial_temp=1.0, cooling=0.995):
    """
    Simulated annealing to improve the solution.
    Moves: swap residues, add a new modulus, remove a modulus.
    """
    current = {m: r for m, r in classes}
    current_score = _score(classes)
    best = {m: r for m, r in classes}
    best_score = current_score
    
    temp = initial_temp
    all_moduli = list(range(20, 201))
    
    while time.time() < deadline and temp > 0.01:
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.5 and len(current) > 0:
            # Swap residues: pick two moduli and swap their residues
            if len(current) < 2:
                temp *= cooling
                continue
            items = list(current.items())
            (m1, r1), (m2, r2) = random.sample(items, 2)
            
            # Try swapping
            new_current = dict(current)
            new_current[m1] = r2
            new_current[m2] = r1
            
            # Check validity
            valid = True
            for m, r in new_current.items():
                if m == m1 or m == m2:
                    continue
                for m2_, r2_ in [(m1, r2), (m2, r1)]:
                    g = gcd(m, m2_)
                    if (r - r2_) % g == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_score = _score([[m, r] for m, r in new_current.items()])
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current = new_current
                    current_score = new_score
                    if current_score > best_score:
                        best = dict(current)
                        best_score = current_score
                        best_classes = [[m, r] for m, r in best.items()]
            
        elif move_type < 0.8:
            # Add a new modulus
            available = [m for m in all_moduli if m not in current]
            if available:
                m = random.choice(available)
                # Find a valid residue
                valid_r = None
                for r in range(m):
                    if _is_valid_addition(m, r, current):
                        valid_r = r
                        break
                if valid_r is not None:
                    new_current = dict(current)
                    new_current[m] = valid_r
                    new_score = _score([[m, r] for m, r in new_current.items()])
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                        current = new_current
                        current_score = new_score
                        if current_score > best_score:
                            best = dict(current)
                            best_score = current_score
                            best_classes = [[m, r] for m, r in best.items()]
        
        else:
            # Remove a modulus
            if current:
                m_to_remove = random.choice(list(current.keys()))
                new_current = dict(current)
                del new_current[m_to_remove]
                new_score = _score([[m, r] for m, r in new_current.items()])
                delta = new_score - current_score
                # Removal always decreases score, so only accept if it enables better future moves
                # For simplicity, only accept with very low probability
                if random.random() < pow(2.718281828, delta / temp) * 0.1:
                    current = new_current
                    current_score = new_score
        
        temp *= cooling
    
    return [[m, r] for m, r in best.items()], best_score

def _final_polish(classes, deadline):
    """
    Final polish: systematically try all residues for each modulus
    to see if a better assignment exists.
    """
    current = {m: r for m, r in classes}
    current_score = _score(classes)
    
    # Sort by 1/m descending (m ascending) to prioritize high-value moduli
    sorted_moduli = sorted(current.keys())
    
    for m in sorted_moduli:
        if time.time() > deadline:
            break
        
        # Try all residues for this modulus
        best_r = current[m]
        best_score = current_score
        
        for r in range(m):
            if r == current[m]:
                continue
            
            # Check if changing this residue maintains validity
            valid = True
            for m2, r2 in current.items():
                if m2 == m:
                    continue
                g = gcd(m, m2)
                if (r - r2) % g == 0:
                    valid = False
                    break
            
            if valid:
                # Compute new score (only this modulus changes, but score depends only on m, not r)
                # So score doesn't change with residue! But it might enable adding more moduli.
                # For now, just keep the first valid residue
                pass
        
        # After trying all residues, try to add any missing moduli
        all_moduli = list(range(20, 201))
        for m_new in all_moduli:
            if time.time() > deadline:
                break
            if m_new in current:
                continue
            
            for r_new in range(m_new):
                if _is_valid_addition(m_new, r_new, current):
                    current[m_new] = r_new
                    current_score = _score([[m, r] for m, r in current.items()])
                    break
    
    return [[m, r] for m, r in current.items()], current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial greedy solution
    best_classes = _greedy_fill([], deadline)
    best_score = _score(best_classes)
    
    # Simulated annealing phase
    sa_deadline = deadline - 10
    if time.time() < sa_deadline:
        sa_classes, sa_score = _simulate_annealing(best_classes, sa_deadline)
        if sa_score > best_score:
            best_classes = sa_classes
            best_score = sa_score
    
    # Final polish phase
    polish_deadline = deadline - 2
    if time.time() < polish_deadline:
        polished_classes, polished_score = _final_polish(best_classes, polish_deadline)
        if polished_score > best_score:
            best_classes = polished_classes
            best_score = polished_score
    
    return best_classes
