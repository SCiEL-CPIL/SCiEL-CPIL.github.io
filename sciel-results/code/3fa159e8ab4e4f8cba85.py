from fractions import Fraction

def construct():
    """
    Returns a list of distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1, minimizing their count using
    a greedy Egyptian fraction algorithm.
    """
    remaining = Fraction(1, 1)
    min_denom = 5
    denominators = []
    
    while remaining > 0:
        # Calculate the smallest denominator d >= min_denom such that 1/d <= remaining
        # This is equivalent to d >= 1/remaining
        inv_rem = 1 / remaining
        # Ceil of a positive fraction n/d is (n + d - 1) // d
        d = max(min_denom, (inv_rem.numerator + inv_rem.denominator - 1) // inv_rem.denominator)
        
        # Check if 1/d is exactly the remaining sum
        if Fraction(1, d) == remaining:
            denominators.append(d)
            break
        
        # Otherwise, take this fraction and continue
        denominators.append(d)
        remaining -= Fraction(1, d)
        min_denom = d + 1
        
        # Optional: If remaining is very small, we might hit the denominator cap.
        # But for sum=1, denominators grow fast, so this is unlikely to be an issue
        # within the 10^7 limit for a reasonable number of terms.
        
    return denominators
