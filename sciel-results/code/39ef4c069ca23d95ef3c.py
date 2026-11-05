"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        return s
    
    def abs_frac(f):
        return abs(f)
    
    best_sum = None
    best_coeffs = None
    
    def try_local_search(start_coeffs):
        nonlocal best_sum, best_coeffs
        if time.time() > deadline:
            return
        coeffs = list(start_coeffs)
        current_sum = compute_sum(coeffs)
        if current_sum == 0:
            # Need nonzero; try a small perturbation
            for i in range(30):
                old = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val != old:
                        coeffs[i] = new_val
                        s = compute_sum(coeffs)
                        if s != 0:
                            current_sum = s
                            break
                coeffs[i] = old
                if current_sum != 0:
                    break
            if current_sum == 0:
                # Force a nonzero
                coeffs[0] = 1
                current_sum = compute_sum(coeffs)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = compute_sum(coeffs)
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    # Accept if it reduces absolute value
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                else:
                    continue
                break
            # Update global best
            if best_sum is None or abs(current_sum) < abs(best_sum):
                best_sum = current_sum
                best_coeffs = list(coeffs)
        
        # Final update
        if best_sum is None or abs(current_sum) < abs(best_sum):
            best_sum = current_sum
            best_coeffs = list(coeffs)
    
    # Try several random starts
    random.seed(42)
    num_starts = 200
    for trial in range(num_starts):
        if time.time() > deadline:
            break
        # Random start: mostly zeros with some ±1
        coeffs = []
        for i in range(30):
            r = random.random()
            if r < 0.3:
                coeffs.append(1)
            elif r < 0.6:
                coeffs.append(-1)
            else:
                coeffs.append(0)
        try_local_search(coeffs)
    
    # Also try structured starts
    # Start with all 1s then flip
    for start_type in range(5):
        if time.time() > deadline:
            break
        if start_type == 0:
            coeffs = [1] * 30
        elif start_type == 1:
            coeffs = [-1] * 30
        elif start_type == 2:
            coeffs = [1 if i % 2 == 0 else -1 for i in range(30)]
        elif start_type == 3:
            coeffs = [1 if i % 2 == 0 else 0 for i in range(30)]
        else:
            coeffs = [0] * 30
            coeffs[0] = 1
            coeffs[1] = -1
        try_local_search(coeffs)
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
