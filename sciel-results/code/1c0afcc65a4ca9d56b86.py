"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1,
    minimizing their count.
    
    Uses a greedy Egyptian fraction algorithm with forced denominator lower bounds (starting at 5)
    to iteratively select the largest possible unit fraction that does not exceed the remaining sum,
    ensuring distinct denominators and minimizing term count via standard greedy optimality heuristics.
    """
    from fractions import Fraction
    
    # We want to represent 1 as a sum of distinct unit fractions with denominators >= 5
    remaining = Fraction(1, 1)
    min_denom = 5
    denominators = []
    
    while remaining > 0:
        if remaining == 0:
            break
        
        # The next denominator must be at least min_denom
        # For a unit fraction 1/d to not exceed remaining, we need 1/d <= remaining
        # So d >= 1/remaining
        # Also d >= min_denom
        # We want the largest unit fraction (smallest d) that doesn't exceed remaining
        # So we want the smallest d >= max(min_denom, ceil(1/remaining))
        
        # Compute ceil(1/remaining)
        # 1/remaining = remaining.denominator / remaining.numerator
        inv_num = remaining.denominator
        inv_den = remaining.numerator
        # ceil(inv_num / inv_den)
        lower_bound = (inv_num + inv_den - 1) // inv_den
        
        # The denominator must be at least max(min_denom, lower_bound)
        d = max(min_denom, lower_bound)
        
        # Check that 1/d <= remaining
        # 1/d <= remaining  <=>  1 <= remaining * d  <=>  d >= 1/remaining
        # We already ensured d >= ceil(1/remaining), so 1/d <= remaining should hold
        # But let's verify and adjust if needed
        while Fraction(1, d) > remaining:
            d += 1
        
        # Also ensure d is not already used (distinct denominators)
        while d in denominators:
            d += 1
            # Re-check that 1/d <= remaining
            while Fraction(1, d) > remaining:
                d += 1
        
        denominators.append(d)
        remaining -= Fraction(1, d)
        min_denom = d + 1  # Next denominator must be strictly larger to ensure distinctness
        
        # Safety check: if we've exceeded the max number of terms
        if len(denominators) > 256:
            break
        
        # Safety check: if remaining is too small to be represented with a single term
        # and we can't make progress, break
        if remaining > 0 and len(denominators) >= 256:
            break
    
    # Verify the result
    if len(denominators) < 1 or len(denominators) > 256:
        raise RuntimeError("invalid number of denominators")
    
    total = Fraction(0, 1)
    for d in denominators:
        if d < 5 or d > 10000000:
            raise RuntimeError(f"denominator {d} out of range")
        total += Fraction(1, d)
    
    if total != 1:
        raise RuntimeError(f"sum is {total}, not 1")
    
    if len(set(denominators)) != len(denominators):
        raise RuntimeError("denominators not distinct")
    
    return denominators
# EVOLVE-BLOCK-END
