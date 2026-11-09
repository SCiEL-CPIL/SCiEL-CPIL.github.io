"""
Direct lookup for Egyptian fraction of 5/31 with minimal largest denominator.
Uses an integer factorization approach for 3-term solutions to exhaustively
search for any solution with max denominator < 248, and falls back to the
known solution [8, 31, 248] if none is found within the time limit.
"""

from fractions import Fraction
import time
import math

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256


def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET


def _find_3term_with_max_c(c, deadline):
    """
    Find 3 distinct denominators a < b < c with 1/a + 1/b + 1/c = 5/31.
    Uses the fact that 1/a + 1/b = 5/31 - 1/c = (5c - 31) / (31c).
    Then (a+b)/(ab) = (5c-31)/(31c), so ab(5c-31) = 31c(a+b).
    Rearranging: (5c-31)a - 31c a = 31c b - (5c-31)b ... wait, let's be careful.
    
    1/a + 1/b = N/D where N = 5c - 31, D = 31c.
    Then (a+b)/(ab) = N/D => D(a+b) = N ab => N ab - D a - D b = 0.
    Multiply by N: N^2 ab - ND a - ND b = 0.
    Add D^2: (Na - D)(Nb - D) = D^2.
    So we need to find divisors x of D^2 such that x = Na - D and D^2/x = Nb - D.
    Then a = (x + D) / N and b = (D^2/x + D) / N.
    We need a and b to be positive integers, distinct, and a < b < c.
    """
    N = 5 * c - 31
    D = 31 * c
    if N <= 0:
        return None
    
    D2 = D * D
    
    # Iterate through divisors x of D^2
    # x = Na - D, so x > 0 => Na > D => a > D/N = 31c/(5c-31).
    # Since a < c, we have 31c/(5c-31) < c => 31c < 5c^2 - 31c => 5c^2 - 62c > 0 => c > 12.4.
    # For c >= 13, this holds.
    
    # We need x to be a divisor of D^2.
    # x = Na - D => a = (x + D) / N must be an integer.
    # So x + D must be divisible by N.
    # x = k * N - D for some integer k? No, x is a divisor.
    # Let's just iterate divisors.
    
    # To find divisors of D^2, we can iterate up to sqrt(D^2) = D.
    # But D can be large. c is up to 247, so D = 31*247 = 7657. D^2 is ~58 million.
    # Iterating up to 7657 is fast.
    
    # Actually, we can be smarter.
    # x must be a divisor of D^2.
    # Also a = (x+D)/N must be integer, so x = m*N - D for some m.
    # And x must divide D^2.
    # So m*N - D | D^2.
    
    # Let's just iterate x from 1 to sqrt(D^2) and check if x divides D^2.
    # If so, check the other divisor y = D^2 // x.
    # For each divisor x, check if (x + D) % N == 0. If so, a = (x+D)//N.
    # Then check if (y + D) % N == 0. If so, b = (y+D)//N.
    # Ensure a < b < c and a != b.
    
    sqrt_D2 = int(math.isqrt(D2))
    
    # We want a < b, so x < y, which means x < sqrt(D^2).
    # So we iterate x up to sqrt_D2.
    
    for x in range(1, sqrt_D2 + 1):
        if time.time() > deadline:
            return None
        
        if D2 % x != 0:
            continue
        
        y = D2 // x
        
        # x corresponds to a, y corresponds to b (since x < y implies a < b? 
        # a = (x+D)/N, b = (y+D)/N. Since x < y, a < b.)
        
        if (x + D) % N != 0:
            continue
        
        a = (x + D) // N
        
        if (y + D) % N != 0:
            continue
        
        b = (y + D) // N
        
        # Check constraints: a < b < c, a, b distinct, positive integers.
        if a < 1 or b < 1:
            continue
        if a >= b:
            continue
        if b >= c:
            continue
        if a == b:
            continue
        
        return [a, b, c]
    
    return None


def _search_3term_exhaustive(max_c, deadline):
    """
    Search for any 3-term solution with max denominator <= max_c.
    Iterates c from 13 to max_c.
    """
    for c in range(13, max_c + 1):
        if time.time() > deadline:
            return None
        sol = _find_3term_with_max_c(c, deadline)
        if sol and _verify(sol):
            return sol
    return None


def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Known solution
    known = [8, 31, 248]
    if _verify(known):
        best = known
        best_max = 248
    else:
        best = None
        best_max = MAX_DENOM + 1
    
    # Search for 3-term solutions with c < 248
    # This is now an exhaustive search using the factorization method.
    # It's fast because for each c, we only iterate up to sqrt((31c)^2) = 31c.
    # Sum_{c=13}^{247} 31c ~ 31 * (247*248/2 - 13*12/2) ~ 31 * 30500 ~ 945500 iterations.
    # This should be very fast in Python.
    
    sol = _search_3term_exhaustive(247, deadline)
    if sol:
        best = sol
        best_max = max(sol)
    
    if best is not None:
        return best
    
    # Fallback: greedy Egyptian fraction decomposition
    r = TARGET
    result = []
    last_d = 0
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= last_d:
            d = last_d + 1
        result.append(d)
        r -= Fraction(1, d)
        last_d = d
    return result
