"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for the 60s wall-clock limit
    
    one = Fraction(1)
    
    # Pre-verified solution using greedy (always valid, may not be minimal)
    # But first, try to find a minimal solution via search
    
    def search(remaining, min_den, terms_left, current):
        """Search for exactly terms_left more terms to sum to remaining.
        Returns list of denominators or None."""
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            return list(current) if remaining == 0 else None
        
        if terms_left == 1:
            # 1/d = remaining => d = 1/remaining
            if remaining > 0:
                d_frac = one / remaining
                if d_frac.denominator == 1:
                    d = d_frac.numerator
                    if d >= min_den and d >= 5 and d <= 10000000:
                        current.append(d)
                        return list(current)
                        current.pop()
            return None
        
        # Lower bound on d: 1/d <= remaining => d >= ceil(1/remaining)
        # Also d >= min_den
        if remaining <= 0:
            return None
        
        # d must satisfy 1/d <= remaining, so d >= 1/remaining
        # ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
        d_lower = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound: we need terms_left terms, each 1/d_i with d_i >= d
        # The max sum with terms_left terms starting at d is sum(1/(d+i) for i in range(terms_left))
        # But we need this to be >= remaining
        # Also, for a given d, the remaining after picking d is remaining - 1/d
        # which must be achievable with terms_left-1 terms starting from d+1
        
        # Pruning: max possible sum with terms_left terms from min_den
        # sum(1/min_den + i for i in range(terms_left))
        max_possible = sum(Fraction(1, min_den + i) for i in range(terms_left))
        if remaining > max_possible:
            return None
        
        # For each candidate d:
        # new_remaining = remaining - 1/d
        # Must be achievable with terms_left-1 terms from d+1
        # Max achievable with terms_left-1 terms from d+1: sum(1/(d+1+i) for i in range(terms_left-1))
        # So we need new_remaining <= that max
        # Also new_remaining >= 0
        
        # Upper bound on d: 
        # We need remaining - 1/d >= 0 => d >= 1/remaining (already have this)
        # And remaining - 1/d <= max_rest(d) where max_rest(d) = sum(1/(d+1+i) for i in range(terms_left-1))
        # This gives a constraint on d
        
        # Simple upper bound: d can't be too large or 1/d becomes too small
        # If d is very large, 1/d is tiny, and we need terms_left-1 terms to sum to ~remaining
        # But the max sum with terms_left-1 terms from d+1 is approximately (terms_left-1)/(d+1)
        # So we need remaining - 1/d <= (terms_left-1)/(d+1)
        # This gives: remaining <= 1/d + (terms_left-1)/(d+1)
        # For large d, RHS ~ terms_left/d, so d <= terms_left/remaining (roughly)
        
        d_upper = min(10000000, int(terms_left / remaining) + 100)
        if d_upper < d_lower:
            return None
        
        for d in range(d_lower, d_upper + 1):
            if d < 5:
                continue
            val = Fraction(1, d)
            if val > remaining:
                continue
            
            new_remaining = remaining - val
            if new_remaining < 0:
                continue
            
            if new_remaining == 0:
                if terms_left == 1:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                # Can't use more terms if remaining is 0
                continue
            
            # Check if new_remaining is achievable with terms_left-1 terms from d+1
            # Max possible: sum(1/(d+1+i) for i in range(terms_left-1))
            if terms_left - 1 > 0:
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(terms_left - 1))
                if new_remaining > max_rest:
                    break  # For larger d, max_rest only decreases
            
            current.append(d)
            result = search(new_remaining, d + 1, terms_left - 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Iterative deepening: try 2 terms, 3 terms, ..., up to some limit
    # With denominators >= 5, 1 term is impossible (1/5 < 1)
    # 2 terms: 1/a + 1/b = 1 => (a-1)(b-1) = ab => only 1/2+1/2=1 but not distinct and <5
    # So minimum is likely 5 or more
    
    best_result = None
    
    for max_terms in range(2, 20):
        if time.time() > deadline - 5:
            break
        result = search(one, 5, max_terms, [])
        if result is not None:
            best_result = result
            break
    
    # Fallback: greedy algorithm (always terminates, gives valid but possibly non-minimal solution)
    if best_result is None:
        remaining = one
        result = []
        min_d = 5
        for _ in range(256):
            if time.time() > deadline - 2:
                break
            if remaining <= 0:
                break
            if remaining == 0:
                break
            # Find smallest d >= min_d such that 1/d <= remaining
            d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            while True:
                val = Fraction(1, d)
                if val <= remaining:
                    result.append(d)
                    remaining -= val
                    min_d = d + 1
                    break
                d += 1
            if remaining == 0:
                break
            if len(result) >= 256:
                break
        
        if len(result) >= 1:
            best_result = result
    
    if best_result is not None:
        return best_result
    
    # Last resort: hardcoded known valid solution
    # Greedy from 5: 1/5, then 1/6, 1/12, 1/18, 1/36, ...
    # Let me just use a verified set
    # 1 = 1/5 + 1/6 + 1/30 + 1/10 + 1/15 ... no
    # Use the greedy result as last resort
    remaining = one
    result = []
    min_d = 5
    for _ in range(256):
        if remaining == 0:
            break
        d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        val = Fraction(1, d)
        if val > remaining:
            d += 1
            val = Fraction(1, d)
        result.append(d)
        remaining -= val
        min_d = d + 1
    
    if len(result) >= 1:
        return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
