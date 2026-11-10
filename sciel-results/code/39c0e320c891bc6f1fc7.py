"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    best_max = [float('inf')]
    best_solution = [None]
    
    def search(remaining, current_max, denominators_used, depth):
        if not remaining:
            if current_max < best_max[0]:
                best_max[0] = current_max
                best_solution[0] = list(denominators_used)
            return
        
        if depth > 8:
            return
        
        num = remaining.numerator
        den = remaining.denominator
        d_min = (den + num - 1) // num
        
        for d in range(d_min, best_max[0]):
            if d in denominators_used:
                continue
            
            frac_d = Fraction(1, d)
            if frac_d > remaining:
                continue
            
            if frac_d == remaining:
                new_max = max(current_max, d)
                if new_max < best_max[0]:
                    best_max[0] = new_max
                    best_solution[0] = list(denominators_used) + [d]
                return
            
            new_remaining = remaining - frac_d
            new_max = max(current_max, d)
            if new_max >= best_max[0]:
                break
            
            search(new_remaining, new_max, denominators_used | {d}, depth + 1)
    
    search(target, 0, frozenset(), 0)
    
    if best_solution[0] is None:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution[0]
# EVOLVE-BLOCK-END
