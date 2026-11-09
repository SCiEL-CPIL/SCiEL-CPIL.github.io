from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000

def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET

def _find_best_3_term_solution(limit_max_denom):
    """
    Find the 3-term Egyptian fraction 1/a + 1/b + 1/c = 5/31
    with max(a,b,c) < limit_max_denom.
    Returns the solution with the smallest max denominator, or None.
    """
    best_max = limit_max_denom
    best_sol = None
    
    # a must be in [7, 18]
    for a in range(7, 19):
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        den_sq = den * den
        
        # We need (num*b - den)(num*c - den) = den^2
        # Let x = num*b - den, y = num*c - den
        # x*y = den^2
        # b = (x+den)/num, c = (y+den)/num
        
        # Iterate divisors x of den_sq
        # x <= sqrt(den_sq)
        x = 1
        while x * x <= den_sq:
            if den_sq % x == 0:
                y = den_sq // x
                
                # Check pair (x, y)
                b_num = x + den
                c_num = y + den
                if b_num % num == 0 and c_num % num == 0:
                    b = b_num // num
                    c = c_num // num
                    if a < b < c and c < best_max:
                        if _verify([a, b, c]):
                            best_max = c
                            best_sol = [a, b, c]
                
                # Check pair (y, x) if x != y
                if x != y:
                    b_num = y + den
                    c_num = x + den
                    if b_num % num == 0 and c_num % num == 0:
                        b = b_num // num
                        c = c_num // num
                        if a < b < c and c < best_max:
                            if _verify([a, b, c]):
                                best_max = c
                                best_sol = [a, b, c]
            x += 1
            
    return best_sol

def construct():
    # Known solution with max denom 248
    best_sol = [8, 31, 248]
    best_max = 248
    
    # Search for 3-term solutions with max denom < 248
    sol = _find_best_3_term_solution(best_max)
    if sol:
        best_sol = sol
        best_max = max(sol)
        
    return best_sol
