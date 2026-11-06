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
    1. Precompute GCD table for speed.
    2. Multi-start greedy initialization.
    3. Efficient Simulated Annealing.
    4. Local search polish.
    """
    deadline = time.time() + 148.0
    random.seed(42)
    
    # Precompute GCD table for moduli 20..200
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    gcd_table = [[0]*n_mods for _ in range(n_mods)]
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            gcd_table[i][j] = gcd(m1, m2)
            
    # Map modulus to index for fast lookup
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        idx_m = mod_to_idx[m]
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                idx_m2 = mod_to_idx[m2]
                g = gcd_table[idx_m][idx_m2]
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
    
    def greedy_init(order):
        classes = []
        for m in order:
            if time.time() > deadline - 10.0:
                break
            new_classes = try_add(classes, m)
            if new_classes is not None:
                classes = new_classes
        return classes
    
    # Multi-start: Try different greedy orders
    best_classes = []
    best_score = 0.0
    
    orders = [
        sorted(moduli, key=lambda m: -1.0/m),
        sorted(moduli, key=lambda m: -1.0/m, reverse=True), # Largest first?
        moduli[:], # Randomized
        sorted(moduli, key=lambda m: -len([i for i in range(2, m) if m % i == 0])) # Most divisors first
    ]
    
    # Shuffle the random one
    random.shuffle(orders[2])
    
    for order in orders:
        if time.time() > deadline - 20.0:
            break
        classes = greedy_init(order)
        score = calc_score(classes)
        if score > best_score:
            best_score = score
            best_classes = classes[:]
            
    # Simulated Annealing
    def sa_polish(classes, temperature=1.0, max_steps=5000):
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        
        step = 0
        while step < max_steps:
            if time.time() > deadline - 10.0:
                break
            step += 1
            
            move_type = random.random()
            
            if move_type < 0.25:
                # Residue reassignment
                if len(current) == 0: continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r: continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                valid = True
                for j in range(len(new_classes)):
                    if j == i: continue
                    m2, r2 = new_classes[j]
                    g = gcd_table[mod_to_idx[m]][mod_to_idx[m2]]
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid: continue
                delta = 0.0
            elif move_type < 0.55:
                # Add
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing: continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None: continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.75:
                # Remove
                if len(current) <= 2: continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            else:
                # Swap
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing or len(current) == 0: continue
                m_add = random.choice(missing)
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None: continue
                new_classes.append([m_add, r])
                delta = 1.0/m_add - 1.0/m_remove
            
            new_score = current_score + delta
            
            if new_score > current_score or random.random() < math.exp(delta / temperature):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            
            temperature *= 0.9998
        
        return best, best_score
    
    # Run SA multiple times from the best start
    for _ in range(5):
        if time.time() > deadline - 15.0:
            break
        sa_result, sa_score = sa_polish(best_classes, temperature=0.5, max_steps=3000)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score

    # Final Hill Climb
    def hill_climb(classes, time_limit=5.0):
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            current_score = calc_score(classes)
            
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

    if time.time() < deadline - 5.0:
        polished = hill_climb(best_classes, time_limit=5.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score

    return best_classes
# EVOLVE-BLOCK-END
