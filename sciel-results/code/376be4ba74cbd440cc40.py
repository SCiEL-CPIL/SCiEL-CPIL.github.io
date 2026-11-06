"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """Find 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1, minimizing count."""
    
    def find_egyptian(remaining, min_denom, max_terms, path):
        """Find representation of 'remaining' as sum of at most max_terms unit fractions
        with denominators >= min_denom, strictly increasing. Returns list of denominators or None."""
        if remaining == 0:
            return list(path)
        if max_terms == 0:
            return None
        if remaining > 1:
            return None
        
        # Minimum denominator: d must satisfy 1/d <= remaining, so d >= ceil(1/remaining)
        # Also d >= min_denom
        num, den = remaining.numerator, remaining.denominator
        if num == 0:
            return list(path)
        # ceil(den/num)
        min_d = (den + num - 1) // num
        min_d = max(min_d, min_denom)
        
        # Maximum denominator: with max_terms terms, each at most 1/min_d, 
        # we need sum >= remaining, so max_terms/min_d >= remaining
        # => min_d <= max_terms / remaining = max_terms * den / num
        # Also for the current term: 1/d <= remaining => d >= 1/remaining (already handled)
        # For remaining terms to work: (max_terms-1) * (1/(d+1)) >= remaining - 1/d
        # This gives an upper bound on d. Let's use: d <= max_terms / remaining
        max_d = max_terms * den // num
        if max_d < min_d:
            max_d = min_d
        max_d = min(max_d, 10000000)
        
        # Try each possible denominator
        for d in range(min_d, max_d + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                return list(path) + [d]
            if max_terms - 1 >= 1:
                result = find_egyptian(new_remaining, d + 1, max_terms - 1, path + [d])
                if result is not None:
                    return result
        return None
    
    target = Fraction(1, 1)
    
    # Try increasing number of terms. We know 2 and 3 terms are impossible with denom >= 5.
    # Start from 4 terms.
    for num_terms in range(4, 30):
        result = find_egyptian(target, 5, num_terms, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
