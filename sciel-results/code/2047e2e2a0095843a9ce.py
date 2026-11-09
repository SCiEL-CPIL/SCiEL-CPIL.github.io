import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # seconds, leave 5s margin
    
    moduli = list(range(2, 129))
    
    best_solution = []
    best_len = 0
    
    def is_compatible(r_new, m_new, existing):
        for r_old, m_old in existing:
            g = math.gcd(m_new, m_old)
            if g == 1:
                return False
            if (r_new - r_old) % g == 0:
                return False
        return True

    def backtrack(idx, current):
        nonlocal best_solution, best_len
        
        if time.time() > deadline:
            return
            
        if len(current) + (len(moduli) - idx) <= best_len:
            return
            
        if idx == len(moduli):
            if len(current) > best_len:
                best_len = len(current)
                best_solution = [pair.copy() for pair in current]
            return
            
        m = moduli[idx]
        
        # Try to include m
        # Iterate residues
        for r in range(m):
            if is_compatible(r, m, current):
                current.append([r, m])
                backtrack(idx + 1, current)
                current.pop()
        
        # Try to skip m
        backtrack(idx + 1, current)

    backtrack(0, [])
    
    if not best_solution:
        return [[0, 2], [1, 4]]

    return best_solution
