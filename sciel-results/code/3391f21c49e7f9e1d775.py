"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from fractions import Fraction
from math import gcd

def construct():
    """
    Find 30 coefficients from {-1,0,1} minimizing |sum c_i / i| with sum != 0.
    Uses randomized search with local optimization.
    """
    n = 30
    
    # We want to minimize |sum_{i=1}^{30} c_i / i| where c_i in {-1, 0, 1}
    # and the sum is nonzero.
    
    # The minimum possible nonzero absolute value is 1/LCM(1..30)
    # But that's astronomically small. In practice, we want to get as close to 0 as possible.
    
    # Strategy: Use a greedy/local search approach.
    # 1. Start with a random configuration
    # 2. Try to reduce the absolute value by flipping or changing individual coefficients
    # 3. Use simulated annealing or hill climbing
    
    # Better approach: Use the fact that we can pair terms to cancel.
    # For example, 1/2 - 1/3 + 1/6 = 0, but we need nonzero.
    
    # Let's use a randomized local search:
    best_val = None
    best_coeffs = None
    
    # Try multiple random restarts
    for attempt in range(200):
        # Initialize with random coefficients
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Ensure the sum is nonzero
        current_sum = sum(Fraction(c, i+1) for i, c in enumerate(coeffs))
        if current_sum == 0:
            # Flip one coefficient to make it nonzero
            for i in range(n):
                if coeffs[i] != 1:
                    coeffs[i] += 1
                else:
                    coeffs[i] = 0
                current_sum = sum(Fraction(c, i+1) for i, c in enumerate(coeffs))
                if current_sum != 0:
                    break
        
        if current_sum == 0:
            continue
        
        # Local search: try to improve by changing one coefficient at a time
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for new_c in [-1, 0, 1]:
                    if new_c == coeffs[i]:
                        continue
                    old_c = coeffs[i]
                    coeffs[i] = new_c
                    new_sum = sum(Fraction(c, j+1) for j, c in enumerate(coeffs))
                    if new_sum == 0:
                        coeffs[i] = old_c
                        continue
                    if best_val is None or abs(new_sum) < abs(best_val):
                        if best_val is None or abs(new_sum) < abs(best_val):
                            best_val = new_sum
                            best_coeffs = coeffs[:]
                        coeffs[i] = old_c
                        improved = True
                        break
                if improved:
                    break
        
        # Also try the current best from this iteration
        if best_val is None or abs(current_sum) < abs(best_val):
            best_val = current_sum
            best_coeffs = coeffs[:]
    
    if best_coeffs is None:
        # Fallback: simple nonzero configuration
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
