"""Find Egyptian fraction of 5/31 minimizing the largest denominator."""
# EVOLVE-BLOCK-START
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000

def construct():
    # Known solution: 5/31 = 1/8 + 1/28 + 1/1736
    # Max denominator: 1736
    # Distinct: Yes
    # Range: Yes
    candidate = [8, 28, 1736]
    
    # Verify
    s = sum(Fraction(1, d) for d in candidate)
    if s == TARGET and all(1 <= d <= MAX_DENOM for d in candidate) and len(set(candidate)) == len(candidate):
        return candidate
    
    # Fallback to a safe, valid solution if the hardcoded one fails (it shouldn't)
    # Greedy algorithm
    r = TARGET
    greedy_vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d > MAX_DENOM:
            break
        greedy_vals.append(d)
        r -= Fraction(1, d)
        if len(greedy_vals) >= 256:
            break
    
    if r == 0 and all(1 <= d <= MAX_DENOM for d in greedy_vals):
        return greedy_vals
        
    # Last resort: return a single term if possible (not possible for 5/31)
    # Or an empty list which is invalid. 
    # But 5/31 is not 1/N.
    # We must return a valid list.
    # If the hardcoded one failed, we have a problem.
    # But we verified it.
    return candidate
# EVOLVE-BLOCK-END
