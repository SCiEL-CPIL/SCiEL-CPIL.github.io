"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def is_compatible(r_new, m_new, chosen):
        for r, m in chosen:
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True
    
    best_result = []
    best_count = 0
    
    random.seed(42)
    
    # Phase 1: Greedy with multiple orderings
    orderings = []
    orderings.append(sorted(all_moduli))
    orderings.append(sorted(all_moduli, reverse=True))
    orderings.append(sorted(all_moduli, key=lambda m: -m.bit_count()))
    
    for _ in range(300):
        order = all_moduli[:]
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 15:
            break
        chosen = []
        for m in order:
            if time.time() > deadline - 15:
                break
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(r, m, chosen):
                    chosen.append((r, m))
                    break
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = chosen[:]
    
    # Phase 2: Try to add more moduli to the best result
    if time.time() < deadline - 5:
        chosen_mod_set = set(m for r, m in best_result)
        remaining = [m for m in all_moduli if m not in chosen_mod_set]
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for m in remaining[:]:
                if time.time() > deadline - 5:
                    break
                for r in range(m):
                    if is_compatible(r, m, best_result):
                        best_result.append((r, m))
                        remaining.remove(m)
                        best_count += 1
                        improved = True
                        break
    
    if best_count > 0:
        return [[r, m] for r, m in best_result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
