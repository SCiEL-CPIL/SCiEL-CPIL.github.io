"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def is_compatible(r_new, m_new, assigned):
        """assigned: dict m -> r"""
        for m, r in assigned.items():
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True
    
    def solve_csp(moduli, time_limit):
        """Backtracking CSP with MRV and forward checking."""
        if time.time() > time_limit:
            return None
        
        # Start with empty assignment
        assigned = {}
        
        def backtrack(remaining):
            if time.time() > time_limit:
                return False
            
            if not remaining:
                return True
            
            # MRV: pick modulus with fewest compatible residues
            best_m = None
            best_residues = None
            best_count = float('inf')
            
            for m in remaining:
                residues = [r for r in range(m) if is_compatible(r, m, assigned)]
                count = len(residues)
                if count == 0:
                    return False  # Dead end
                if count < best_count:
                    best_count = count
                    best_m = m
                    best_residues = residues
                    if count == 1:
                        break
            
            # Try residues for best_m
            random.shuffle(best_residues)
            for r in best_residues:
                assigned[best_m] = r
                new_remaining = [m for m in remaining if m != best_m]
                if backtrack(new_remaining):
                    return True
                del assigned[best_m]
            
            return False
        
        if backtrack(moduli):
            return [[assigned[m], m] for m in moduli]
        return None
    
    def greedy_build(moduli_order, time_limit):
        chosen = {}
        for m in moduli_order:
            if time.time() > time_limit:
                break
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(r, m, chosen):
                    chosen[m] = r
                    break
        return [[chosen[m], m] for m in chosen]
    
    def local_search(chosen, time_limit):
        """Try to add moduli or swap residues."""
        chosen = {m: r for r, m in chosen}
        chosen_mods = set(chosen.keys())
        available = [m for m in all_moduli if m not in chosen_mods]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add each available modulus
            for m in available[:]:
                if m in chosen_mods:
                    continue
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(r, m, chosen):
                        chosen[m] = r
                        chosen_mods.add(m)
                        improved = True
                        break
            
            # Try swapping residues
            if not improved:
                for m in list(chosen.keys()):
                    if time.time() >= time_limit:
                        break
                    old_r = chosen[m]
                    for r_new in range(m):
                        if r_new == old_r:
                            continue
                        # Check compatibility with all others
                        ok = True
                        for m2, r2 in chosen.items():
                            if m2 == m:
                                continue
                            g = gcd(m, m2)
                            if r_new % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            # Try to add an available modulus
                            for m_add in available[:]:
                                if m_add in chosen_mods:
                                    continue
                                for r_add in range(m_add):
                                    # Check with all including new m
                                    ok2 = True
                                    for m3, r3 in chosen.items():
                                        if m3 == m:
                                            if r_add % gcd(m_add, m) == r_new % gcd(m_add, m):
                                                ok2 = False
                                                break
                                        else:
                                            if r_add % gcd(m_add, m3) == r3 % gcd(m_add, m3):
                                                ok2 = False
                                                break
                                    if ok2:
                                        chosen[m] = r_new
                                        chosen[m_add] = r_add
                                        chosen_mods.add(m_add)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if not improved:
                                chosen[m] = old_r
                            else:
                                break
                    if improved:
                        break
        
        return [[chosen[m], m] for m in chosen]
    
    best_result = []
    best_count = 0
    
    random.seed(42)
    
    # Phase 1: Multiple greedy passes
    for trial in range(100):
        if time.time() > deadline - 20:
            break
        moduli_order = all_moduli[:]
        random.shuffle(moduli_order)
        chosen = greedy_build(moduli_order, deadline - 20)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = chosen[:]
    
    # Phase 2: Local search on best results
    if time.time() < deadline - 5:
        improved = local_search(best_result, deadline - 5)
        if len(improved) > best_count:
            best_count = len(improved)
            best_result = improved
    
    # Phase 3: Try CSP solver on subsets
    if time.time() < deadline - 10:
        # Try to solve for all moduli (will likely fail fast, but try)
        # Actually, try on the best set plus some extras
        chosen_mods = set(m for _, m in best_result)
        available = [m for m in all_moduli if m not in chosen_mods]
        
        # Try adding a few at a time with CSP
        for extra_count in [5, 10, 15]:
            if time.time() > deadline - 10:
                break
            sample = random.sample(available, min(extra_count, len(available)))
            test_moduli = list(chosen_mods) + sample
            random.shuffle(test_moduli)
            
            # Try CSP on this set
            result = solve_csp(test_moduli, deadline - 10)
            if result and len(result) > best_count:
                best_count = len(result)
                best_result = result
    
    # Phase 4: More local search
    if time.time() < deadline - 3:
        improved = local_search(best_result, deadline - 3)
        if len(improved) > best_count:
            best_count = len(improved)
            best_result = improved
    
    if best_count > 0:
        return [[r, m] for r, m in best_result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
