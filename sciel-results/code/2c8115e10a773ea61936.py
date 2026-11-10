import time
from fractions import Fraction

def _verify(dens):
    """Verify that denominators are distinct, in range, and sum to 1."""
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    if any(d < 5 or d > 10000000 for d in dens):
        return False
    s = sum(Fraction(1, d) for d in dens)
    return s == 1

def _find_solution_8(min_den=5, max_den=10000000, deadline=None):
    """
    Find 8 distinct denominators >= min_den whose reciprocal sum is exactly 1.
    Strategy: pick 6 denominators, solve for last 2.
    """
    if deadline is None:
        deadline = time.time() + 40
    
    target = Fraction(1, 1)
    
    def try_solve_last2(dens):
        """Try to solve for last 2 given 6 denominators."""
        current_sum = sum(Fraction(1, d) for d in dens)
        if current_sum >= target:
            return None
        R = target - current_sum
        
        min_a = dens[-1] + 1 if dens else min_den
        # We need 1/a + 1/b = R where a > b > a_min
        # So 1/a < R, meaning a > 1/R
        # And 1/b > R, meaning b < 1/R
        # More precisely: 1/a + 1/b = R => b = a/(Ra - 1)
        # For b > a: a/(Ra-1) > a => 1/(Ra-1) > 1 => Ra-1 < 1 => Ra < 2 => a < 2/R
        # So a is in (1/R, 2/R)
        
        low_a = max(min_a, int(1 / R) + 1)
        high_a = min(int(2 / R) - 1, max_den)
        
        if high_a < low_a:
            return None
        
        seen = set(dens)
        
        for a in range(low_a, high_a + 1):
            Ra_minus_1 = R * a - 1
            if Ra_minus_1 <= 0:
                continue
            b_frac = Fraction(a, 1) / Ra_minus_1
            if b_frac.denominator == 1:
                b = b_frac.numerator
                if b > a and b <= max_den and a not in seen and b not in seen:
                    return dens + [a, b]
        return None
    
    def search(dens, idx):
        if time.time() > deadline:
            return None
        
        if idx == 6:
            return try_solve_last2(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        # Bounding: the maximum additional sum from remaining terms
        remaining_terms = 6 - idx
        if remaining_terms > 0:
            # Maximum sum from 'remaining_terms' terms starting from 'start'
            # is 1/start + 1/(start+1) + ... + 1/(start + remaining_terms - 1)
            max_possible = sum(Fraction(1, start + i) for i in range(remaining_terms))
            current_sum = sum(Fraction(1, d) for d in dens)
            if current_sum + max_possible < target:
                return None
        
        end = min(start + 200, max_den)
        
        for d in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            current_sum = sum(Fraction(1, x) for x in dens) + Fraction(1, d)
            if current_sum >= target:
                break
            
            result = search(dens + [d], idx + 1)
            if result is not None:
                return result
        
        return None
    
    return search([], 0)

def construct():
    deadline = time.time() + 55
    
    # First, try to find an 8-term solution
    res = _find_solution_8(deadline=deadline)
    if res is not None and _verify(res):
        return res
    
    # Fallback: 11-term solution
    res = _find_solution_11(deadline=deadline)
    if res is not None and _verify(res):
        return res
    
    # Hard fallback: known 7-term solution
    known = [5, 6, 7, 10, 14, 35, 70]
    if _verify(known):
        return known
    
    raise RuntimeError("no valid representation found")

def _find_solution_11(min_den=5, max_den=10000000, deadline=None):
    """
    Find 11 distinct denominators >= min_den whose reciprocal sum is exactly 1.
    Strategy: pick 9 denominators, solve for last 2.
    """
    if deadline is None:
        deadline = time.time() + 40
    
    target = Fraction(1, 1)
    
    def try_solve(dens):
        """Try to solve for last 2 given 9 denominators."""
        current_sum = sum(Fraction(1, d) for d in dens)
        if current_sum >= target:
            return None
        R = target - current_sum
        
        min_a = dens[-1] + 1 if dens else min_den
        low_a = max(min_a, int(1 / R) + 1)
        high_a = min(int(2 / R) - 1, max_den)
        
        if high_a < low_a:
            return None
        
        seen = set(dens)
        
        for a in range(low_a, high_a + 1):
            Ra_minus_1 = R * a - 1
            if Ra_minus_1 <= 0:
                continue
            b_frac = Fraction(a, 1) / Ra_minus_1
            if b_frac.denominator == 1:
                b = b_frac.numerator
                if b > a and b <= max_den and a not in seen and b not in seen:
                    return dens + [a, b]
        return None
    
    def search(dens, idx):
        if time.time() > deadline:
            return None
        
        if idx == 9:
            return try_solve(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        remaining_terms = 9 - idx
        if remaining_terms > 0:
            max_possible = sum(Fraction(1, start + i) for i in range(remaining_terms))
            current_sum = sum(Fraction(1, d) for d in dens)
            if current_sum + max_possible < target:
                return None
        
        end = min(start + 150, max_den)
        
        for d in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            current_sum = sum(Fraction(1, x) for x in dens) + Fraction(1, d)
            if current_sum >= target:
                break
            
            result = search(dens + [d], idx + 1)
            if result is not None:
                return result
        
        return None
    
    return search([], 0)
