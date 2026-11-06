"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55  # 5s safety margin before 60s kill
    
    # Try known solutions with fewer terms first, then search
    # Best known: 5 terms? Let's check if 5 terms with denominators >=5 exists.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.2+0.1667+0.1429+0.125+0.1111 = 0.7457... not 1
    # Need larger denominators to get closer to 1 with fewer terms? No, smaller denominators give larger reciprocals.
    # With 5 terms, max sum is 1/5+1/6+1/7+1/8+1/9 ≈ 0.746 < 1. So 5 terms impossible.
    # With 6 terms: max is 1/5+1/6+1/7+1/8+1/9+1/10 ≈ 0.872 < 1. Still impossible.
    # With 7 terms: max is 1/5+1/6+1/7+1/8+1/9+1/10+1/11 ≈ 0.963 < 1. Still impossible.
    # With 8 terms: max is 1/5+...+1/12 ≈ 1.031 >= 1. So 8 terms might work.
    # Let's check: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 = 0.2+0.16667+0.14286+0.125+0.11111+0.1+0.09091+0.08333 = 1.01988 > 1
    # So we need to find 8 denominators in [5, 10000000] summing reciprocals to exactly 1.
    
    # The current solution has 10 terms. Can we do 8 or 9?
    # Let's use a backtracking search to find 8-term solutions.
    
    def search(n_terms, remaining, min_denom, current):
        if time.time() > deadline:
            return None
        if len(current) == n_terms - 1:
            # Last term must be exactly remaining
            if remaining > 0:
                d = remaining.numerator // remaining.denominator
                if d == remaining.numerator / remaining.denominator and d >= min_denom and d >= 5 and d <= 10000000:
                    if d.is_integer() and int(d) == d:
                        return current + [int(d)]
            return None
        if len(current) >= n_terms:
            return None
        
        # Pruning: even with the largest possible remaining reciprocals, can we reach 1?
        # Max additional sum with (n_terms - len(current)) terms starting from min_denom
        # This is complex, so we'll use a simpler bound
        
        # Find the next denominator
        # The next denominator d must satisfy:
        # 1/d < remaining (otherwise sum would exceed 1)
        # So d > 1/remaining
        max_d = int(1 / remaining) if remaining > 0 else 10000000
        min_d = max(min_denom, 5)
        
        # Also, we need enough remaining terms to reach the target
        # If we pick d, the new remaining is remaining - 1/d
        # We need remaining - 1/d to be achievable with (n_terms - 1 - len(current)) more terms
        
        for d in range(min_d, min(max_d + 1, 10000001)):
            if time.time() > deadline:
                return None
            if d in current:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break  # d too small, 1/d too large
            
            # Pruning: can we achieve new_remaining with remaining terms?
            # Max sum with k terms starting from d+1 is at most 1/(d+1) + 1/(d+2) + ... 
            # But this is hard to bound tightly. Let's just try.
            
            result = search(n_terms, new_remaining, d + 1, current + [d])
            if result is not None:
                return result
        
        return None
    
    # Try 8 terms first
    for n in [8, 9]:
        result = search(n, Fraction(1, 1), 5, [])
        if result is not None and len(result) == n:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(d >= 5 and d <= 10000000 for d in result) and len(set(result)) == n:
                return sorted(result)
    
    # Fallback: return the known 10-term solution
    fallback = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback
    
    # Ultimate fallback
    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
# EVOLVE-BLOCK-END
