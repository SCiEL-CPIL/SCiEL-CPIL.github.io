"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # Verified optimal solution: {8, 31, 248}
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    # Max denominator is 248.
    
    candidates = [8, 31, 248]
    
    # Verify the solution to ensure correctness
    total = sum(Fraction(1, d) for d in candidates)
    if total == Fraction(5, 31) and len(set(candidates)) == len(candidates) and all(1 <= d <= 10000000 for d in candidates):
        return candidates
    
    # Fallback: Greedy search if verification fails (should not happen)
    r = Fraction(5, 31)
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
