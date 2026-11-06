"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    deadline = time.time() + 85  # 85s wall-clock guard (90s limit, 5s margin)
    
    def greedy_egyptian(target_frac, min_denom, max_denom=10000000):
        """Greedy Egyptian fraction with strictly increasing denominators >= min_denom."""
        result = []
        remaining = target_frac
        last_denom = min_denom - 1
        
        while remaining > 0:
            if time.time() > deadline:
                return None
            
            # Find the smallest d >= last_denom + 1 such that 1/d <= remaining
            # i.e., d >= 1/remaining
            min_d = max(last_denom + 1, min_denom)
            # d must satisfy: 1/d <= remaining => d >= 1/remaining
            # Since remaining = p/q, 1/remaining = q/p
            # So d >= ceil(q/p)
            
            if remaining.numerator == 0:
                break
                
            # d >= ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d = max(lower, min_d)
            
            # Ensure d is valid
            if d > max_denom:
                return None
            
            # Check if 1/d <= remaining
            # 1/d <= remaining => remaining.numerator * d >= remaining.denominator
            if remaining.numerator * d < remaining.denominator:
                d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
                d = max(d, min_d)
                if d > max_denom:
                    return None
            
            result.append(d)
            remaining = remaining - Fraction(1, d)
            last_denom = d
            
            if len(result) > 256:
                return None
        
        if remaining == 0:
            return result
        return None
    
    # Start with the greedy approach
    solution = greedy_egyptian(Fraction(1, 1), 5)
    
    if solution is not None:
        return solution
    
    # Fallback: try the DP approach with a smaller target
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
