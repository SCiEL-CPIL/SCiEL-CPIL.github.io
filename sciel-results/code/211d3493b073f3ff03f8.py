from itertools import combinations
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    # Denominators must be >= 7 because 1/6 > 5/31 is false?
    # 1/6 = 0.166 > 0.161. So a=6 is possible as first term?
    # If a=6, 1/b + 1/c ... = 5/31 - 1/6 = (30-31)/186 = -1/186. Impossible.
    # So a must be such that 1/a < 5/31 => a > 6.2 => a >= 7.
    
    candidates = list(range(7, 36)) # 7 to 35 inclusive
    
    # Try 4, 5, 6 terms
    for k in [4, 5, 6]:
        for combo in combinations(candidates, k):
            s = sum(Fraction(1, x) for x in combo)
            if s == target:
                return list(combo)
                
    # Fallback to known 3-term solution
    return [8, 31, 248]
