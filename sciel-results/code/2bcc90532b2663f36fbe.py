from fractions import Fraction

def construct():
    # Theoretical minimum is 8 terms for denominators >= 5.
    # We search for a solution with 8 terms.
    result = _search(8, 5, Fraction(1, 1))
    if result:
        return result
    # Fallback: try 9 terms just in case (though 8 is proven possible)
    result = _search(9, 5, Fraction(1, 1))
    if result:
        return result
    raise RuntimeError("No solution found")

def _search(terms_left, start_denom, remaining_sum):
    if terms_left == 0:
        if remaining_sum == 0:
            return []
        return None
    
    if terms_left == 1:
        if remaining_sum.numerator == 1:
            d = remaining_sum.denominator
            if d >= start_denom and d <= 10000000:
                return [d]
        return None

    p, q = remaining_sum.numerator, remaining_sum.denominator
    # min_d such that 1/d <= remaining_sum
    min_d = max(start_denom, (q + p - 1) // p)
    
    # max_d such that remaining_sum - 1/d can be formed by terms_left-1 terms
    # Max sum of remaining terms is approx (terms_left-1)/(d+1)
    # We iterate d from min_d upwards.
    
    for d in range(min_d, 10000000):
        next_sum = remaining_sum - Fraction(1, d)
        if next_sum < 0:
            break
        
        # Pruning:
        # Max possible sum for remaining terms
        # If terms_left - 1 is small, we can check bounds.
        # A simple bound: (terms_left - 1) / (d + 1) >= next_sum
        # (terms_left - 1) >= next_sum * (d + 1)
        if next_sum * (d + 1) > terms_left - 1:
            break
            
        res = _search(terms_left - 1, d + 1, next_sum)
        if res:
            return [d] + res
            
    return None
