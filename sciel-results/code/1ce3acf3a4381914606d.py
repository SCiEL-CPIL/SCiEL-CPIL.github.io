"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Use a greedy Egyptian fraction algorithm with forced denominator lower bounds (starting at 5)
    to iteratively select the largest possible unit fraction that does not exceed the remaining sum,
    ensuring distinct denominators and minimizing term count.
    """
    remaining = Fraction(1, 1)
    denominators = []
    min_denom = 5  # denominators must be >= 5
    
    while remaining > 0:
        # We need to find the smallest denominator d >= min_denom such that 1/d <= remaining
        # i.e., d >= 1/remaining
        # Also, we must ensure d is not already in denominators
        
        # Lower bound for d from the constraint 1/d <= remaining:
        # d >= ceil(1 / remaining)
        
        # Compute 1 / remaining as a fraction
        inv_remaining = Fraction(1, 1) / remaining
        # inv_remaining is a Fraction. We need d >= inv_remaining.
        # So the smallest integer d satisfying this is ceil(inv_remaining).
        
        # But we also have the constraint d >= min_denom
        lower_bound = max(min_denom, -(-inv_remaining.numerator // inv_remaining.denominator))  # ceil division
        
        # Now find the smallest d >= lower_bound that is not in denominators
        # and such that 1/d <= remaining (which is guaranteed by d >= ceil(1/remaining))
        
        d = lower_bound
        denom_set = set(denominators)
        while d in denom_set:
            d += 1
        
        # Check that 1/d <= remaining
        # Since d >= ceil(1/remaining), we have d >= 1/remaining, so 1/d <= remaining.
        # But let's be safe and verify:
        if Fraction(1, d) > remaining:
            # This shouldn't happen if our logic is correct, but just in case
            # Increase d until 1/d <= remaining
            while Fraction(1, d) > remaining:
                d += 1
                while d in denom_set:
                    d += 1
        
        # Select d
        denominators.append(d)
        denom_set.add(d)
        remaining -= Fraction(1, d)
        min_denom = d + 1  # Next denominator must be strictly greater than d
        
        # Safety check: if remaining is still positive but we've exceeded reasonable bounds
        if d > 10000000:
            raise RuntimeError("Denominator exceeded limit")
    
    # Verify the result
    if not denominators:
        raise RuntimeError("No denominators found")
    
    # Check distinctness
    if len(denominators) != len(set(denominators)):
        raise RuntimeError("Denominators are not distinct")
    
    # Check range
    for d in denominators:
        if d < 5 or d > 10000000:
            raise RuntimeError(f"Denominator {d} out of range [5, 10000000]")
    
    # Check count
    if len(denominators) < 1 or len(denominators) > 256:
        raise RuntimeError(f"Number of denominators {len(denominators)} out of range [1, 256]")
    
    # Verify sum is exactly 1
    total = sum(Fraction(1, d) for d in denominators)
    if total != 1:
        raise RuntimeError(f"Sum is {total}, not 1")
    
    return denominators

# EVOLVE-BLOCK-END
