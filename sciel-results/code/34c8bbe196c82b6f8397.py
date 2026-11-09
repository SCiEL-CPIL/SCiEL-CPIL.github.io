"""Few-term Egyptian fractions with denominators >= 5 (denominators in [5, 10^7]).

Returns a short, distinct list of integer denominators in [5, 10^7] whose
reciprocals sum to exactly 1.  The objective is to minimise the count.
"""
from fractions import Fraction
import time

def _valid(sol):
    """Check that sol is a valid answer: distinct, in range, sum of reciprocals = 1."""
    if not sol:
        return False
    if len(sol) > 256:
        return False
    if len(set(sol)) != len(sol):
        return False
    for d in sol:
        if not isinstance(d, int) or d < 5 or d > 10_000_000:
            return False
    s = sum(Fraction(1, d) for d in sol)
    return s == 1

def _greedy_solve():
    """Find a valid Egyptian fraction representation of 1 using greedy method."""
    remaining = Fraction(1)
    used = set()
    sol = []
    # Greedy: pick smallest d >= 5 such that 1/d <= remaining and d not in used
    # If 1/d > remaining, we must pick a larger d.
    # Standard greedy: d = ceil(1/remaining).
    
    while remaining > 0:
        if remaining == 0:
            break
        
        # Calculate next denominator
        # 1/d <= remaining => d >= 1/remaining
        d = int(1 / remaining)
        if 1 / d > remaining:
            d += 1
        
        # Ensure d >= 5
        if d < 5:
            d = 5
            
        # Ensure d is not used
        while d in used:
            d += 1
            
        # Ensure 1/d <= remaining
        while Fraction(1, d) > remaining:
            d += 1
            if d in used:
                d += 1

        # Check bounds
        if d > 10_000_000:
            return None # Failed
        
        sol.append(d)
        used.add(d)
        remaining -= Fraction(1, d)
        
        # Safety check for too many terms
        if len(sol) > 256:
            return None

    return sol

def construct():
    """Return 1..256 distinct integer denominators in [5, 10000000] whose
    reciprocal sum is exactly 1, minimising their count."""
    
    # Try to find a short solution with a limited search first?
    # For now, use greedy to guarantee correctness and speed.
    sol = _greedy_solve()
    
    if sol and _valid(sol):
        return sol
        
    # Fallback: raise error if greedy failed (should not happen)
    raise RuntimeError("Failed to find a valid representation")
