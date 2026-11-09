import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin for 90s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    # Precompute gcd for all pairs
    gcd_cache = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            gcd_cache[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        key = (min(m1, m2), max(m1, m2))
        return gcd_cache.get(key, math.gcd(m1, m2))
    
    def is_compatible(r1, m1, r2, m2):
        """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
        g = get_gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def find_valid_residues(m, selected):
        """Find all residues r for modulus m that are compatible with all selected."""
        valid = []
        for r in range(m):
            compatible = True
            for m_p, r_p in selected:
                if not is_compatible(r, m, r_p, m_p):
                    compatible = False
                    break
            if compatible:
                valid.append(r)
        return valid
    
    # Phase 1: Dynamic Greedy with Most-Constrained-First Heuristic
    best_solution = []
    
    def dynamic_greedy(randomize_ties=False, seed=None):
        selected = []
        remaining = set(moduli)
        rng = random.Random(seed) if seed is not None else random
        
        while remaining:
            if time.time() > deadline:
                break
            
            # For each remaining modulus, find valid residues
            # and count them. We want to pick the one with the fewest valid residues
            # (Most Constrained First). If tie, prefer primes, then smaller moduli.
            
            candidates = []
            for m in remaining:
                valid_residues = find_valid_residues(m, selected)
                if valid_residues:
                    # Heuristic score: 
                    # 1. Minimize number of valid residues (most constrained)
                    # 2. Prefer primes (less likely to cause future conflicts)
                    # 3. Prefer smaller moduli
                    score = (len(valid_residues), 0 if m in primes else 1, m)
                    candidates.append((score, m, valid_residues))
            
            if not candidates:
                break
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x[0])
            
            # Pick the best candidate
            min_count = candidates[0][0][0]
            tied_candidates = [c for c in candidates if c[0][0] == min_count]
            
            if randomize_ties and len(tied_candidates) > 1:
                chosen = rng.choice(tied_candidates)
            else:
                chosen = tied_candidates[0]
            
            _, m, valid_residues = chosen
            r = rng.choice(valid_residues) if randomize_ties else valid_residues[0]
            
            selected.append((m, r))
            remaining.remove(m)
            
        return selected
    
    # Run dynamic greedy with various randomization settings
    # 1. Deterministic (pick first in sorted order, first residue)
    sol = dynamic_greedy(randomize_ties=False)
    if len(sol) > len(best_solution):
        best_solution = sol
    
    # 2. Randomized ties and residues - many restarts
    for i in range(200):
        if time.time() > deadline:
            break
        sol = dynamic_greedy(randomize_ties=True, seed=i)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Phase 2: Local search - try to add more moduli
    for _ in range(100):  # Iterations
        if time.time() > deadline:
            break
        
        improved = False
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        # Try to add each unselected modulus
        for m in unselected_mods:
            if time.time() > deadline:
                break
            valid_residues = find_valid_residues(m, best_solution)
            if valid_residues:
                r = random.choice(valid_residues) if len(valid_residues) > 1 else valid_residues[0]
                best_solution.append((m, r))
                improved = True
                break
        
        # Try changing residues of selected moduli to allow adding more
        if not improved:
            for i in range(len(best_solution)):
                if time.time() > deadline:
                    break
                m, r = best_solution[i]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    # Check if new_r is compatible with all other selected
                    compatible = True
                    for j in range(len(best_solution)):
                        if j == i:
                            continue
                        m_p, r_p = best_solution[j]
                        if not is_compatible(new_r, m, r_p, m_p):
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Try adding an unselected modulus with this change
                    test_solution = best_solution.copy()
                    test_solution[i] = (m, new_r)
                    
                    for m_add in unselected_mods:
                        valid_add = find_valid_residues(m_add, test_solution)
                        if valid_add:
                            r_add = random.choice(valid_add) if len(valid_add) > 1 else valid_add[0]
                            best_solution = test_solution + [(m_add, r_add)]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    # Phase 3: More aggressive local search - try swapping residues
    for _ in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        if not unselected_mods:
            break
        
        # Try removing one selected modulus and adding two unselected
        for i in range(len(best_solution)):
            if time.time() > deadline:
                break
            
            m_remove, r_remove = best_solution[i]
            reduced_solution = best_solution[:i] + best_solution[i+1:]
            
            # Try adding two unselected moduli
            added_count = 0
            new_solution = reduced_solution.copy()
            for m_add in unselected_mods:
                if time.time() > deadline:
                    break
                if m_add in set(m for m, r in new_solution):
                    continue
                valid_add = find_valid_residues(m_add, new_solution)
                if valid_add:
                    r_add = random.choice(valid_add) if len(valid_add) > 1 else valid_add[0]
                    new_solution.append((m_add, r_add))
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count > 1:  # Net gain
                best_solution = new_solution
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 4: Simulated annealing style - try random perturbations
    current_solution = best_solution.copy()
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Randomly perturb: remove a random element and try to re-add with different residue
        if len(current_solution) < 2:
            break
        
        idx = random.randint(0, len(current_solution) - 1)
        m, r = current_solution[idx]
        reduced = current_solution[:idx] + current_solution[idx+1:]
        
        valid_residues = find_valid_residues(m, reduced)
        if valid_residues:
            # Try a different residue
            other_residues = [x for x in valid_residues if x != r]
            if other_residues:
                new_r = random.choice(other_residues)
                new_solution = reduced + [(m, new_r)]
                
                # Try to add more moduli to the new solution
                selected_mods = set(mm for mm, _ in new_solution)
                unselected = [mm for mm in moduli if mm not in selected_mods]
                
                for m_add in unselected:
                    valid_add = find_valid_residues(m_add, new_solution)
                    if valid_add:
                        r_add = random.choice(valid_add) if len(valid_add) > 1 else valid_add[0]
                        new_solution.append((m_add, r_add))
                        break
                
                if len(new_solution) > len(current_solution):
                    current_solution = new_solution
                elif random.random() < 0.3:  # Accept worse with some probability
                    current_solution = new_solution
        
        if len(current_solution) > len(best_solution):
            best_solution = current_solution.copy()
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
