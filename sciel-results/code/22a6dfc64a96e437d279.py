import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 30
    
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    # Integer weights: w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Initialize with random coefficients in {-1, 0, 1}
    # Ensure non-zero sum
    def random_solution():
        while True:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            total = sum(c * w for c, w in zip(coeffs, weights))
            if total != 0:
                return coeffs, total

    best_coeffs, best_total = random_solution()
    best_abs = abs(best_total)
    
    # Split indices into two halves for meet-in-the-middle style local search or just use for partitioning
    half = n // 2
    
    # Local search: coordinate descent with random restarts and simulated annealing
    # We will use a hybrid approach:
    # 1. Fast coordinate descent to converge to a local minimum.
    # 2. Simulated annealing to escape local minima.
    
    def coordinate_descent(coeffs, total):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                old_val = coeffs[i]
                w = weights[i]
                # Try all 3 values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    new_total = total + (new_val - old_val) * w
                    if new_total == 0:
                        continue
                    if abs(new_total) < abs(total):
                        coeffs[i] = new_val
                        total = new_total
                        changed = True
                        break
        return coeffs, total

    # Run initial coordinate descent
    best_coeffs, best_total = coordinate_descent(best_coeffs, best_total)
    best_abs = abs(best_total)
    
    # Simulated Annealing
    T_start = 500.0
    T_end = 0.01
    alpha = 0.99
    
    # Use a better initial temperature based on the current gap
    # The weights are large, so deltas are large.
    # Max weight is L. L for 1..30 is 2329089562800.
    # This is huge. exp(-delta/T) will be 0 for any reasonable T.
    # We need to scale T or use a different acceptance criterion.
    # Let's try scaling T by the magnitude of the weights or the current best_abs.
    
    # Alternative: Use a "plateau" search or random walks with occasional jumps.
    # Given the huge weights, the sum is an integer. The "energy" is |total|.
    # Most moves will increase |total| significantly.
    # We should focus on moves that keep |total| small or slightly increase it.
    
    # Let's try a different strategy:
    # Since L is large, the problem is equivalent to finding a subset sum close to 0.
    # We can use a heuristic based on the magnitude of weights.
    # Sort weights by magnitude? No, indices are fixed.
    
    # Let's try random restarts with coordinate descent.
    # And a "kick" phase where we change a few random coefficients.
    
    while time.time() < deadline - 5.0:
        # Random restart with kick
        coeffs = list(best_coeffs)
        
        # Kick: change 3-5 random coefficients
        num_kicks = random.randint(3, 5)
        for _ in range(num_kicks):
            idx = random.randint(0, n - 1)
            old_val = coeffs[idx]
            options = [v for v in [-1, 0, 1] if v != old_val]
            new_val = random.choice(options)
            coeffs[idx] = new_val
            
        # Recalculate total
        total = sum(c * w for c, w in zip(coeffs, weights))
        
        if total == 0:
            # Invalid, adjust one coeff
            for i in range(n):
                old_val = coeffs[i]
                options = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(options)
                new_total = total + (new_val - old_val) * weights[i]
                if new_total != 0:
                    coeffs[i] = new_val
                    total = new_total
                    break
        
        # Coordinate descent
        coeffs, total = coordinate_descent(coeffs, total)
        
        if abs(total) < best_abs:
            best_abs = abs(total)
            best_total = total
            best_coeffs = coeffs
            
    # Ensure we return native Python ints
    return [int(c) for c in best_coeffs]
