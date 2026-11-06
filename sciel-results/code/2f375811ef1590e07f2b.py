"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    # Try different first denominators and use improved completion
    # First, get a baseline solution
    def greedy_complete(r, used):
        """Complete the expansion using greedy method"""
        if r == 0:
            return []
        if r < 0:
            return None
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in used or d > 10000000:
                # Try next few denominators
                found = False
                for dd in range(d, min(d + 50, 10000001)):
                    if dd not in used and dd <= 10000000:
                        new_r = r - Fraction(1, dd)
                        if 0 <= new_r:
                            values.append(dd)
                            used.add(dd)
                            r = new_r
                            found = True
                            break
                if not found:
                    return None
            else:
                values.append(d)
                used.add(d)
                r -= Fraction(1, d)
        return values
    
    # Try different starting denominators
    for first_d in range(2, 100):
        if time.time() > deadline:
            break
        if first_d > 10000000:
            break
        r = target - Fraction(1, first_d)
        if r < 0:
            continue
        
        used = {first_d}
        completion = greedy_complete(r, used)
        if completion is not None:
            solution = [first_d] + completion
            max_den = max(solution)
            if max_den < best_max_den:
                best_max_den = max_den
                best_solution = solution[:]
                if best_max_den <= 100:  # Good enough, can stop early
                    break
    
    # If no solution found, fall back to original greedy
    if best_solution is None:
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
# EVOLVE-BLOCK-END
