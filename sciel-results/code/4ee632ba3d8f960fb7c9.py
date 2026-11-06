import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 55.0
    
    def check_compatible(r_new, m_new, existing):
        for r_old, m_old in existing:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def greedy_assign(moduli):
        """Deterministic greedy: try residues in natural order."""
        current = []
        for m in moduli:
            found = False
            for r in range(m):
                if check_compatible(r, m, current):
                    current.append((r, m))
                    found = True
                    break
            if not found:
                return None
        return current
    
    best = []
    
    # Strategy 1: Even moduli, increasing
    even_moduli = list(range(2, 129, 2))
    sol = greedy_assign(even_moduli)
    if sol and len(sol) > len(best):
        best = sol
    
    # Strategy 2: Even moduli, decreasing
    sol = greedy_assign(list(reversed(even_moduli)))
    if sol and len(sol) > len(best):
        best = sol
    
    # Strategy 3: Multiples of 4
    mult4 = list(range(4, 129, 4))
    sol = greedy_assign(mult4)
    if sol and len(sol) > len(best):
        best = sol
    
    # Strategy 4: Multiples of 6
    mult6 = list(range(6, 129, 6))
    sol = greedy_assign(mult6)
    if sol and len(sol) > len(best):
        best = sol
    
    # Strategy 5: Randomized restarts on even moduli
    for _ in range(1000):
        if time.time() - start_time > max_time:
            break
        shuffled = list(even_moduli)
        random.shuffle(shuffled)
        current = []
        for m in shuffled:
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if check_compatible(r, m, current):
                    current.append((r, m))
                    found = True
                    break
            if not found:
                break
        if len(current) > len(best):
            best = current
    
    return [[r, m] for r, m in best]
# EVOLVE-BLOCK-END
