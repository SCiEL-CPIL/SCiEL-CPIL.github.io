"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement
import random
import time

def construct():
    best_poly = [1] * 12
    best_nonzero = float('inf')
    
    # Try several structured and random approaches
    attempts = 0
    start_time = time.time()
    
    # Approach 1: Try polynomials with exponents in arithmetic progression
    # and coefficients +/-1, check square sparsity
    for trial in range(5000):
        if time.time() - start_time > 50:
            break
        attempts += 1
        
        # Choose 12 distinct exponents
        # Use a mix of strategies
        if trial % 3 == 0:
            # Arithmetic progression
            base = random.randint(0, 20)
            step = random.randint(1, 5)
            exponents = [base + i * step for i in range(12)]
        elif trial % 3 == 1:
            # Random distinct exponents
            exponents = random.sample(range(0, 60), 12)
        else:
            # Two interleaved arithmetic progressions
            base1 = random.randint(0, 10)
            base2 = random.randint(0, 10)
            step = random.randint(2, 6)
            exponents = sorted([base1 + 2*i*step for i in range(6)] + [base2 + (2*i+1)*step for i in range(6)])
        
        # Ensure distinct and sorted
        exponents = sorted(set(exponents))
        if len(exponents) < 12:
            # Fill in
            while len(exponents) < 12:
                exponents.append(max(exponents) + 1)
            exponents = sorted(exponents[:12])
        
        # Choose coefficients: +/-1
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        
        # Build polynomial as coefficient list
        max_deg = max(exponents)
        if max_deg > 40:
            continue
        
        poly = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        
        # Ensure exactly 12 nonzero terms
        if sum(1 for c in poly if c != 0) != 12:
            continue
        
        # Compute square
        square = np.convolve(poly, poly)
        square = [int(x) for x in square]
        
        # Count nonzero terms
        nonzero = sum(1 for c in square if c != 0)
        
        if nonzero < best_nonzero:
            best_nonzero = nonzero
            best_poly = poly[:]
            if best_nonzero <= 15:  # Good enough, stop early
                break
    
    # Trim trailing zeros
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
    
    # Ensure nonzero final coefficient
    if best_poly[-1] == 0:
        best_poly[-1] = 1
    
    return best_poly
# EVOLVE-BLOCK-END
