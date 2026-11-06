import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 50 # Leave margin
    
    best_max = 1736
    best_sol = [8, 28, 1736]
    
    def dfs(rem, min_d, depth, terms):
        nonlocal best_max, best_sol
        
        if time.time() > deadline:
            return
        
        if rem == 0:
            if terms:
                current_max = terms[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_sol = terms[:]
            return
        
        if depth == 0:
            return
            
        # Pruning
        if rem > Fraction(depth, min_d + 1):
            return
            
        d_min = max(min_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
        d_max = best_max - 1
        
        if d_min > d_max:
            return
            
        for d in range(d_min, d_max + 1):
            # Check if remaining sum can be formed
            # rem - 1/d must be <= (depth-1)/(d+1)
            if rem - Fraction(1, d) > Fraction(depth - 1, d + 1):
                break
                
            terms.append(d)
            dfs(rem - Fraction(1, d), d, depth - 1, terms)
            terms.pop()
            
            if time.time() > deadline:
                return

    for depth in range(2, 20):
        dfs(target, 1, depth, [])
        if time.time() > deadline:
            break
            
    return best_sol
