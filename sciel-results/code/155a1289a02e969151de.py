"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction by value (1/m)
    2. Local search with 2-swap moves
    3. Simulated annealing to escape local optima
    4. Branch-and-bound for final optimization
    """
    deadline = time.time() + 148.0
    random.seed(42)
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    # Initial greedy construction by value (1/m)
    moduli = list(range(20, 201))
    moduli_by_value = sorted(moduli, key=lambda m: -1.0/m)
    classes = []
    for m in moduli_by_value:
        if time.time() > deadline - 10.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    def hill_climb(classes, time_limit=5.0):
        """Hill climbing polish with add, remove, and swap moves."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            current_score = calc_score(classes)
            
            # Try adding
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    new_score = calc_score(new_classes)
                    if new_score > current_score + 1e-12:
                        classes = new_classes
                        current_score = new_score
                        improved = True
            
            # Try swapping (remove one, add another with higher value)
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                for m_remove in current_moduli:
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = calc_score(new_classes)
                        if new_score > current_score + 1e-12:
                            classes = new_classes
                            current_score = new_score
                            improved = True
                            break
        return classes
    
    # Simulated annealing with multiple move types
    def sa_polish(classes, temperature=1.0, max_steps=5000):
        """Simulated annealing with various move types."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        
        step = 0
        while step < max_steps:
            if time.time() > deadline - 10.0:
                break
            step += 1
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.25:
                # Single residue reassignment (no score change)
                if len(current) == 0:
                    continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                # Check validity
                valid = True
                for j in range(len(new_classes)):
                    if j == i:
                        continue
                    m2, r2 = new_classes[j]
                    g = gcd(m, m2)
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid:
                    continue
                delta = 0.0
            elif move_type < 0.55:
                # Add a missing modulus
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing:
                    continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None:
                    continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.75:
                # Remove a modulus
                if len(current) <= 2:
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            else:
                # 2-swap: swap one modulus for another
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing or len(current) == 0:
                    continue
                m_add = random.choice(missing)
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None:
                    continue
                new_classes.append([m_add, r])
                delta = 1.0/m_add - 1.0/m_remove
            
            new_score = current_score + delta
            
            # Acceptance criterion
            if new_score > current_score or random.random() < math.exp(delta / temperature):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            
            # Cool down
            temperature *= 0.9998
        
        return best, best_score
    
    # Branch and bound for final optimization
    def branch_and_bound(classes, time_limit=10.0):
        """Branch and bound to find better solutions."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        start = time.time()
        
        # Try all possible 2-swaps systematically
        current_moduli = set(c[0] for c in current)
        missing = [m for m in moduli if m not in current_moduli]
        
        # Sort missing by value to prioritize high-value moduli
        missing_sorted = sorted(missing, key=lambda m: -1.0/m)
        
        for m_add in missing_sorted:
            if time.time() - start > time_limit:
                break
            for m_remove in current_moduli:
                if 1.0/m_add <= 1.0/m_remove:
                    continue
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is not None:
                    new_classes.append([m_add, r])
                    new_score = calc_score(new_classes)
                    if new_score > best_score + 1e-12:
                        best = new_classes[:]
                        best_score = new_score
                        current = new_classes[:]
                        current_moduli = set(c[0] for c in current)
                        missing = [m for m in moduli if m not in current_moduli]
                        missing_sorted = sorted(missing, key=lambda m: -1.0/m)
        
        return best, best_score
    
    # Run multiple rounds of optimization
    for round_num in range(8):
        if time.time() > deadline - 15.0:
            break
        
        # Polish current best
        polished = hill_climb(best_classes, time_limit=2.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
        
        # SA from current best
        sa_result, sa_score = sa_polish(best_classes, temperature=0.3, max_steps=2000)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Branch and bound
        bb_result, bb_score = branch_and_bound(best_classes, time_limit=3.0)
        if bb_score > best_score:
            best_classes = bb_result
            best_score = bb_score
        
        # Final polish
        polished = hill_climb(best_classes, time_limit=2.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Final hill climb
    if time.time() < deadline - 5.0:
        polished = hill_climb(best_classes, time_limit=5.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    return best_classes
# EVOLVE-BLOCK-END
