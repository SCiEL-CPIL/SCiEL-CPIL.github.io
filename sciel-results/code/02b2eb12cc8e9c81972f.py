"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Use greedy Egyptian fraction algorithm
    # Repeatedly pick the smallest denominator d >= min_d such that 1/d <= remaining
    # This guarantees distinct denominators and termination for rationals
    
    remaining = Fraction(1, 1)
    denominators = []
    min_d = 5
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find smallest d >= min_d such that 1/d <= remaining
        # This means d >= 1/remaining
        # So d_min = max(min_d, ceil(1/remaining))
        
        # Compute ceil(1/remaining)
        # 1/remaining = remaining.denominator / remaining.numerator
        num, den = remaining.numerator, remaining.denominator
        # d must satisfy 1/d <= num/den => d >= den/num
        # So d_min_candidate = ceil(den/num)
        
        if num == 0:
            break
            
        # ceil(den/num)
        d_candidate = (den + num - 1) // num
        d = max(min_d, d_candidate)
        
        # Verify 1/d <= remaining
        if Fraction(1, d) > remaining:
            d += 1
        
        # Add this term
        denominators.append(d)
        remaining -= Fraction(1, d)
        min_d = d + 1  # Next denominator must be larger to ensure distinctness
        
        # Safety: if we exceed 256 terms, stop
        if len(denominators) > 256:
            break
    
    # Verify the solution
    if denominators and len(denominators) >= 1:
        s = sum(Fraction(1, d) for d in denominators)
        if s == 1 and len(set(denominators)) == len(denominators) and all(5 <= d <= 10000000 for d in denominators):
            return denominators
    
    # Fallback: if greedy failed or produced invalid result, try a hardcoded valid solution
    # Let me construct one by hand using known identities
    # 1 = 1/2 + 1/3 + 1/6
    # Need to express each with denominators >= 5 and all distinct
    
    # Alternative: use the greedy result even if it's long, as long as it's valid
    # The greedy algorithm should always produce a valid solution for 1
    
    # If we get here, something went wrong. Return an empty list or raise error
    # But we need to return a valid solution. Let me try a different approach.
    
    # Actually, the greedy algorithm should work. Let me debug by checking if remaining reached 0.
    # If not, it means we hit the deadline or 256 term limit.
    
    # For now, return what we have if it's valid, otherwise raise
    if denominators:
        s = sum(Fraction(1, d) for d in denominators)
        if s == 1 and len(set(denominators)) == len(denominators) and all(5 <= d <= 10000000 for d in denominators):
            return denominators
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END
