"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # 75s guard (90s kill, 15s safety margin)
    
    # Known 10-term solution: [5, 6, 7, 9, 10, 12, 14, 15, 18, 420]
    # Sum of reciprocals = 1 exactly.
    known_solution = [5, 6, 7, 9, 10, 12, 14, 15, 18, 420]
    
    # Verify the known solution
    total = Fraction(0, 1)
    for d in known_solution:
        total += Fraction(1, d)
    assert total == 1, "Known solution is invalid!"
    
    # Try to find a 9-term solution via backtracking search
    # Strategy: search for distinct denominators a1 < a2 < ... < a9 in [5, 10^7]
    # such that sum(1/ai) = 1.
    # 
    # Key insight: for k terms summing to 1 with a1 < a2 < ... < ak:
    #   1/a1 > 1/k => a1 < k
    #   Since a1 >= 5 and we want k=9: a1 < 9, so a1 in {5, 6, 7, 8}
    
    def search(k, min_den, remaining):
        """
        Find k distinct denominators >= min_den summing to 'remaining' (Fraction).
        Returns list of denominators or None.
        """
        if k == 0:
            if remaining == 0:
                return []
            return None
        if k == 1:
            if remaining.numerator == 1 and remaining.denominator >= min_den:
                return [remaining.denominator]
            return None
        
        # Upper bound for the next denominator:
        # 1/d > remaining / k  =>  d < k / remaining
        upper = int(k / remaining) + 1 if remaining > 0 else 10**7 + 1
        upper = min(upper, 10**7)
        
        if min_den > upper:
            return None
        
        # Try denominators from min_den upward
        for d in range(min_den, upper + 1):
            if time.time() > deadline:
                return None
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            
            # Pruning: max sum of k-1 terms each <= 1/(d+1) is (k-1)/(d+1)
            # If this is less than new_remaining, we can't reach the target
            if (k - 1) / (d + 1) < new_remaining:
                # For larger d, new_remaining decreases but (k-1)/(d+1) also decreases.
                # We need to check if it's still possible.
                # Actually, let's compute a tighter bound.
                # The minimum denominator for the remaining k-1 terms is d+1.
                # The maximum possible sum is (k-1)/(d+1).
                # If (k-1)/(d+1) < new_remaining, then even with the smallest
                # possible denominators, we can't reach new_remaining.
                # So we should continue to larger d where new_remaining is smaller.
                # But wait, as d increases, both sides decrease.
                # Let's just continue.
                pass
            
            result = search(k - 1, d + 1, new_remaining)
            if result is not None:
                return [d] + result
        
        return None
    
    # Try to find a 9-term solution
    # a1 must be in {5, 6, 7, 8} (since 1/a1 > 1/9 => a1 < 9, and a1 >= 5)
    best_9 = None
    for a1 in [5, 6, 7, 8]:
        if time.time() > deadline:
            break
        remaining = 1 - Fraction(1, a1)
        result = search(8, a1 + 1, remaining)
        if result is not None:
            best_9 = [a1] + result
            break
    
    if best_9 is not None:
        # Verify
        total = Fraction(0, 1)
        for d in best_9:
            total += Fraction(1, d)
        if total == 1 and len(set(best_9)) == 9 and all(5 <= d <= 10**7 for d in best_9):
            return best_9
    
    return known_solution
# EVOLVE-BLOCK-END
