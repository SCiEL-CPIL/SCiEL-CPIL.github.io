"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, log, exp

def construct():
    start = time.time()
    deadline = start + 55  # 5s safety margin before 60s kill
    
    def harmonic_sum(n, k):
        """Sum of 1/(n+1) + 1/(n+2) + ... + 1/(n+k) approximated."""
        if k <= 0:
            return 0
        # Use integral approximation: sum_{i=1}^{k} 1/(n+i) ≈ ln((n+k)/(n+1))
        # More precise: H(n+k) - H(n) where H is harmonic number
        # For large n, H(n) ≈ ln(n) + 0.5772156649
        # For our purposes, use the simple bound
        return log((n + k) / (n + 1)) + 0.5 / (n + 1) - 0.5 / (n + k)
    
    def can_achieve(remaining, n_terms_left, min_denom):
        """Check if 'remaining' can be achieved with n_terms_left distinct denominators >= min_denom."""
        if n_terms_left == 0:
            return remaining == 0
        if remaining <= 0:
            return False
        if n_terms_left == 1:
            d = remaining.numerator // remaining.denominator
            if remaining.numerator % remaining.denominator == 0:
                d = remaining.numerator // remaining.denominator
                return d >= min_denom and d <= 10000000
            return False
        
        # Max possible sum with n_terms_left terms starting from min_denom
        max_sum = 0
        for i in range(n_terms_left):
            max_sum += 1.0 / (min_denom + i)
        
        if remaining > max_sum:
            return False
        
        # Min possible sum with n_terms_left terms (using largest denominators)
        # This is very small, so usually not a constraint
        # But we need at least n_terms_left distinct values >= min_denom
        # The smallest sum would use the largest denominators, which approach 0
        # So min_sum ≈ 0, which is always < remaining (since remaining > 0)
        
        return True
    
    def search(n_terms, remaining, min_denom, current, depth):
        if time.time() > deadline:
            return None
        
        terms_left = n_terms - depth
        if terms_left == 0:
            return current if remaining == 0 else None
        
        if terms_left == 1:
            # Last term must be exactly remaining
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if num % den == 0:
                    d = num // den
                    if d >= min_denom and d >= 5 and d <= 10000000 and d not in current:
                        return current + [d]
            return None
        
        # Determine valid range for next denominator
        # 1/d < remaining => d > 1/remaining
        # Also need: remaining - 1/d achievable with (terms_left - 1) terms >= d+1
        
        # Lower bound: d >= min_denom
        # Upper bound: 1/d < remaining => d > 1/remaining
        # So d >= floor(1/remaining) + 1 if remaining > 0
        
        if remaining <= 0:
            return None
        
        # d must satisfy 1/d < remaining, so d > 1/remaining
        # d_min from constraint: d >= floor(1/remaining) + 1
        # But also d >= min_denom
        
        inv_rem = 1.0 / float(remaining)
        d_lower = max(min_denom, int(inv_rem) + 1)
        
        # Upper bound from achievability:
        # After picking d, we need remaining - 1/d achievable with (terms_left-1) terms >= d+1
        # Max sum with (terms_left-1) terms starting from d+1 is approx harmonic_sum(d, terms_left-1)
        # So we need remaining - 1/d <= harmonic_sum(d, terms_left-1)
        # i.e., 1/d >= remaining - harmonic_sum(d, terms_left-1)
        
        # Also, we need remaining - 1/d > 0, which is guaranteed by d > 1/remaining
        
        # For the upper bound on d:
        # The smallest achievable sum with (terms_left-1) terms starting from d+1
        # is very small, so this doesn't give a tight upper bound on d
        # But we can use: remaining - 1/d must be > 0, already handled
        # And remaining - 1/d must be <= max possible sum with (terms_left-1) terms
        
        # Let's iterate d from d_lower upward, but cap it reasonably
        # d cannot be too large because 1/d must be significant enough
        
        # Max d: we need remaining - 1/d to be achievable
        # The minimum non-zero sum with (terms_left-1) terms is very small
        # So d can be large, but 1/d must be < remaining
        # d < 1/remaining, so d <= floor(1/remaining)
        
        d_upper = min(int(inv_rem), 10000000)
        
        if d_lower > d_upper:
            return None
        
        # Limit search space for efficiency
        max_search_range = 100000
        if d_upper - d_lower > max_search_range:
            d_upper = d_lower + max_search_range
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            if d in current:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                continue  # d too small
            
            # Check if new_remaining is achievable with (terms_left - 1) terms >= d+1
            if not can_achieve(new_remaining, terms_left - 1, d + 1):
                # If new_remaining is too large, increase d
                # If new_remaining is too small, this d won't work
                continue
            
            result = search(n_terms, new_remaining, d + 1, current + [d], depth + 1)
            if result is not None:
                return result
        
        return None
    
    # Try 8 terms first, then 9
    for n in [8, 9]:
        result = search(n, Fraction(1, 1), 5, [], 0)
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
