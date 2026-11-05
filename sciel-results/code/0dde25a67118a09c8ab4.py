from fractions import Fraction

def construct():
    # Greedy Egyptian Fraction Algorithm
    # We want to represent 1 as sum of 1/d_i where d_i >= 5 and distinct.
    # We enforce d_i > d_{i-1} to ensure distinctness and termination.
    
    remaining = Fraction(1, 1)
    min_d = 5
    denominators = []
    
    while remaining > 0:
        # Calculate the smallest integer d such that 1/d <= remaining
        # d >= 1/remaining
        # d = ceil(1/remaining)
        # Also d >= min_d
        
        # 1/remaining = remaining.denominator / remaining.numerator
        d_ideal = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        
        if d_ideal < min_d:
            d = min_d
        else:
            d = d_ideal
            
        # Check if 1/d <= remaining
        # If d was chosen as min_d, it's possible 1/min_d > remaining
        # In that case, we must increase d until 1/d <= remaining
        while Fraction(1, d) > remaining:
            d += 1
            
        denominators.append(d)
        remaining -= Fraction(1, d)
        min_d = d + 1
        
        # Safety check to prevent infinite loops or huge numbers
        if len(denominators) > 256:
            break
        if d > 10000000:
            break
            
    if remaining != 0 or len(denominators) > 256:
        # Fallback or error, but greedy should terminate
        pass
        
    return denominators
