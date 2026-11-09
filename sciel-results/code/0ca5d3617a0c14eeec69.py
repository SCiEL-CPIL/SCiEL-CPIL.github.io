import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s safety margin for 60s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    def is_prime(n):
        if n < 2: return False
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
    
    # Phase 1: Greedy with multiple orderings
    best_solution = []
    
    def greedy_with_order(order, randomize_residue=False):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_residues = find_valid_residues(m, selected)
            if valid_residues:
                if randomize_residue:
                    r = random.choice(valid_residues)
                else:
                    r = valid_residues[0]
                selected.append((m, r))
        return selected
    
    # Generate various orderings
    orderings = []
    
    # Primes first (they have the least conflicts)
    orderings.append(primes + composites)
    
    # Sort by number of divisors (fewer divisors = less likely to conflict)
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_count = {m: count_divisors(m) for m in moduli}
    orderings.append(sorted(moduli, key=lambda x: div_count[x]))
    orderings.append(sorted(moduli, key=lambda x: -div_count[x]))
    
    # Sort by max gcd with other moduli
    max_gcds = {m: 0 for m in moduli}
    for m in moduli:
        for m2 in moduli:
            if m != m2:
                g = get_gcd(m, m2)
                if g > max_gcds[m]:
                    max_gcds[m] = g
    orderings.append(sorted(moduli, key=lambda x: -max_gcds[x]))
    orderings.append(sorted(moduli, key=lambda x: max_gcds[x]))
    
    # Random orderings
    for _ in range(300):
        mods = moduli.copy()
        random.shuffle(mods)
        orderings.append(mods)
    
    # Try all orderings
    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy_with_order(order, randomize_residue=True)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Phase 2: Local search - try to add more moduli
    for _ in range(50):  # Iterations
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
    for _ in range(30):
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
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
