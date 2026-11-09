"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def is_compatible(r_new, m_new, chosen):
        for r, m in chosen:
            g = gcd(m_new, m)
            if g > 1 and r_new % g == r % g:
                return False
        return True
    
    # Precompute gcd matrix for speed
    n = len(all_moduli)
    idx_to_mod = {m: i for i, m in enumerate(all_moduli)}
    mod_to_idx = {m: i for i, m in enumerate(all_moduli)}
    
    # Precompute gcd for all pairs
    gcd_matrix = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Greedy with multiple orderings to find good initial solution
    def greedy_solve(order):
        chosen = []
        for m in order:
            if time.time() > deadline - 5:
                break
            found = False
            # Try residues in order 0,1,2,...
            for r in range(m):
                if is_compatible(r, m, chosen):
                    chosen.append((r, m))
                    found = True
                    break
            # If no residue works, skip this modulus
        return chosen
    
    # Try several orderings
    orderings = [
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        sorted(all_moduli, key=lambda m: -m.bit_count()),  # more prime factors first
        sorted(all_moduli, key=lambda m: m),
    ]
    
    # Add random orderings
    random.seed(42)
    for _ in range(5):
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_result = []
    best_count = 0
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        result = greedy_solve(order)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result[:]
    
    # Now try to improve with local search / backtracking on the best solution
    # Try to add more moduli or swap residues to fit more
    def try_add_moduli(chosen, remaining_moduli):
        """Try to add moduli from remaining to chosen."""
        improved = False
        for m in remaining_moduli:
            if time.time() > deadline - 5:
                break
            if m in [cm for _, cm in chosen]:
                continue
            for r in range(m):
                if is_compatible(r, m, chosen):
                    chosen.append((r, m))
                    improved = True
                    break
        return improved
    
    # Iteratively try to add more
    chosen = best_result[:]
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        remaining = [m for m in all_moduli if m not in [cm for _, cm in chosen]]
        if not remaining:
            break
        if not try_add_moduli(chosen, remaining):
            break
    
    if len(chosen) > best_count:
        best_count = len(chosen)
        best_result = chosen[:]
    
    # Try a more sophisticated backtracking on a subset of moduli
    # Focus on moduli that share factors (the constraining ones)
    # Group by prime factors
    def prime_factors(m):
        factors = set()
        d = 2
        temp = m
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        return factors
    
    # Identify moduli that are "constraining" (share factors with others)
    # and try backtracking on those with good variable ordering
    
    def backtrack_solve(moduli_list, time_limit=30):
        """Backtracking with constraint propagation."""
        if time.time() > start_time + time_limit:
            return None
        
        # Sort by number of constraints (moduli with more shared factors first)
        def constraint_count(m):
            count = 0
            for other in moduli_list:
                if other != m and gcd(m, other) > 1:
                    count += 1
            return count
        
        sorted_mods = sorted(moduli_list, key=lambda m: -constraint_count(m))
        assignment = {}
        
        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False
            if idx == len(sorted_mods):
                return True
            
            m = sorted_mods[idx]
            # Try residues
            for r in range(m):
                compatible = True
                for j in range(idx):
                    m_j = sorted_mods[j]
                    r_j = assignment[m_j]
                    g = gcd(m, m_j)
                    if g > 1 and r % g == r_j % g:
                        compatible = False
                        break
                
                if compatible:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False
        
        if backtrack(0):
            return [[assignment[m], m] for m in sorted_mods]
        return None
    
    # Try backtracking on the full set of moduli that were in our best solution
    # plus some others, to see if we can do better
    if best_count < len(all_moduli) and time.time() < deadline - 30:
        # Try backtracking on all moduli
        bt_result = backtrack_solve(all_moduli, time_limit=40)
        if bt_result and len(bt_result) > best_count:
            best_count = len(bt_result)
            best_result = bt_result[:]
    
    if best_count > 0:
        return [[r, m] for r, m in best_result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
