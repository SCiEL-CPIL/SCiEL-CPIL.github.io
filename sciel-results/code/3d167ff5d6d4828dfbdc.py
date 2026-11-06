"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    
    n = 30
    weights = [1.0 / i for i in range(1, n + 1)]
    
    best_abs = float('inf')
    best_coeffs = None
    
    def eval_float(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def eval_frac(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        return s
    
    def local_search(coeffs, time_limit):
        nonlocal best_abs, best_coeffs
        coeffs = list(coeffs)
        current_sum = eval_float(coeffs)
        current_abs = abs(current_sum)
        
        # Verify with exact arithmetic
        if current_sum != 0:
            exact = eval_frac(coeffs)
            if exact != 0:
                exact_abs = abs(exact)
                if exact_abs < best_abs:
                    best_abs = exact_abs
                    best_coeffs = list(coeffs)
        
        while time.time() < time_limit:
            improved = False
            best_move = None
            best_new_sum = current_sum
            best_new_abs = current_abs
            
            for i in range(n):
                old_c = coeffs[i]
                for new_c in (-1, 0, 1):
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * weights[i]
                    new_sum = current_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_sum = new_sum
                        best_move = (i, new_c)
            
            if best_move is None:
                break
            
            i, new_c = best_move
            coeffs[i] = new_c
            current_sum = best_new_sum
            current_abs = best_new_abs
            improved = True
            
            # Check exact
            if current_sum != 0:
                exact = eval_frac(coeffs)
                if exact != 0:
                    exact_abs = abs(exact)
                    if exact_abs < best_abs:
                        best_abs = exact_abs
                        best_coeffs = list(coeffs)
            
            if not improved:
                break
        
        return coeffs
    
    # Multiple random restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial configuration
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Also try some structured starts
        if r == 0:
            coeffs = [0] * n
            coeffs[0] = 1
            coeffs[1] = -1
        elif r == 1:
            coeffs = [0] * n
            coeffs[-1] = 1
            coeffs[-2] = -1
        
        local_search(coeffs, min(time.time() + 1.5, deadline - 1))
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
