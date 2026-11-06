"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = Fraction(5, 31)
    max_den_limit = 1735 # We know 1736 works, try to find < 1736
    
    # Fallback solution
    best_solution = [8, 28, 1736]
    
    # Search for solution with max denominator < 1736
    # d1 must be at least 7
    for d1 in range(7, 100):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        # d2 must be at least 2/r1
        min_d2 = (r1.denominator + r1.numerator - 1) // r1.numerator
        if min_d2 > max_den_limit:
            continue
            
        for d2 in range(min_d2, max_den_limit + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            min_d3 = (r2.denominator + r2.numerator - 1) // r2.numerator
            if min_d3 > max_den_limit:
                continue
                
            # We need to see if r2 can be represented by denominators > d2 and <= max_den_limit
            # Greedy check with pruning
            # If r2 < 1/max_den_limit, we need at least 2 terms? No, 1 term is 1/d.
            # If r2 < 1/max_den_limit, then 1/d > r2 for all d <= max_den_limit.
            # So we need multiple terms.
            
            # Simple heuristic: try to complete greedily, but ensure all denominators <= max_den_limit
            # If greedy produces a denominator > max_den_limit, this branch fails.
            
            r = r2
            current_den = d2
            terms = [d1, d2]
            valid = True
            
            while r > 0:
                min_d = (r.denominator + r.numerator - 1) // r.numerator
                if min_d > max_den_limit:
                    valid = False
                    break
                # Try the smallest possible denominator
                d_next = min_d
                if d_next <= current_den:
                    valid = False
                    break
                terms.append(d_next)
                current_den = d_next
                r -= Fraction(1, d_next)
                
                if len(terms) > 10: # Prevent infinite loops or huge lists
                    valid = False
                    break
            
            if valid and r == 0:
                # Found a solution with max denominator <= max_den_limit
                best_solution = terms
                return best_solution
                
    return best_solution
# EVOLVE-BLOCK-END
