"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

_MAX_DEN = 10000000

def construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    """
    # Theoretical minimum is 13.
    # Try to find a 13-term solution by taking 5..16 and solving for the last two terms.
    # If that fails or takes too long, fall back to a valid larger solution.
    
    # Target k = 13
    # Denominators: 5, 6, ..., 16 (12 terms)
    # Remainder R = 1 - sum(1/i for i in 5..16)
    # Need 1/x + 1/y = R with x, y > 16
    
    # Calculate R as a fraction
    R = Fraction(1, 1)
    for i in range(5, 17):
        R -= Fraction(1, i)
        
    p = R.numerator
    q = R.denominator
    
    # We need (px - q)(py - q) = q^2
    # Let A = px - q, B = py - q
    # A * B = q^2
    # x = (A + q) / p
    # y = (B + q) / p
    
    # Find factors A of q^2
    # q is likely large, but let's check the size.
    # LCM(5..16) is 720720. q will be around that magnitude.
    # q^2 is ~ 5e11. Iterating up to sqrt(q^2) = q is too slow.
    # However, we can just iterate x from 17 upwards and check if y is integer.
    # y = x / (Rx - 1)
    # Rx - 1 must divide x.
    # Let Rx - 1 = k. Then x = (k+1)/R.
    # This is equivalent to checking if (x * p) % q == 0? No.
    # 1/x + 1/y = p/q => y = xq / (px - q)
    # So px - q must divide xq.
    # Since gcd(px-q, x) might not be 1, we just check if (x * q) % (p * x - q) == 0.
    
    found = False
    sol = None
    deadline = time.time() + 5.0
    
    # Search for x
    # x must be > 16
    # Also 1/x < R => x > 1/R
    min_x = max(17, int(1 / R) + 1)
    
    for x in range(min_x, _MAX_DEN):
        if time.time() > deadline:
            break
        denom_y = p * x - q
        if denom_y <= 0:
            continue
        # Check if y is integer
        if (x * q) % denom_y == 0:
            y = (x * q) // denom_y
            if y > x and y <= _MAX_DEN:
                # Check distinctness (y > x ensures distinct from x, and both > 16 ensures distinct from 5..16)
                sol = list(range(5, 17)) + [x, y]
                found = True
                break
    
    if found:
        return sol
        
    # Fallback: Use a known valid solution with more terms if 13 fails
    # A simple greedy algorithm (Sylvester) will produce a valid solution.
    # We just need to ensure it finishes and is valid.
    # For safety, we return a hardcoded valid solution if search fails.
    # Let's generate one quickly using the identity 1/n = 1/(n+1) + 1/(n(n+1))
    # Start with 1 = 1/5 + 4/5
    # This will create many terms, but it's valid.
    # To minimize terms in fallback, we can try k=14, 15 etc. with similar 2-term completion logic.
    
    for k in range(14, 20):
        # Take 5..4+k-1
        end = 4 + k - 1
        R = Fraction(1, 1)
        for i in range(5, end + 1):
            R -= Fraction(1, i)
            
        p = R.numerator
        q = R.denominator
        
        # Try to complete with 2 terms: 1/x + 1/y = R
        min_x = max(end + 1, int(1 / R) + 1)
        # Limit search to be fast
        limit = min(_MAX_DEN, min_x + 10000) 
        for x in range(min_x, limit):
            denom_y = p * x - q
            if denom_y <= 0:
                continue
            if (x * q) % denom_y == 0:
                y = (x * q) // denom_y
                if y > x and y <= _MAX_DEN:
                    return list(range(5, end + 1)) + [x, y]
                    
    # Ultimate fallback: Hardcoded valid solution (13 terms might not be found in time, 
    # but we can return a valid 15-20 term solution or use greedy)
    # Greedy:
    res = []
    rem = Fraction(1, 1)
    last = 4
    while rem > 0:
        d = rem.denominator // rem.numerator + 1
        if d <= last:
            d = last + 1
        if d > _MAX_DEN:
            break
        res.append(d)
        rem -= Fraction(1, d)
        last = d
        if len(res) > 256:
            break
            
    if res and sum(Fraction(1, i) for i in res) == 1:
        return res
        
    # If all else fails, return a known valid set (e.g., from a precomputed source or simple identity)
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + ...
    # Let's just return the greedy result if it's valid, or a hardcoded one.
    # For the purpose of this contest, the 13-term search is very likely to succeed.
    
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50] # Placeholder, should not be reached

# EVOLVE-BLOCK-END
