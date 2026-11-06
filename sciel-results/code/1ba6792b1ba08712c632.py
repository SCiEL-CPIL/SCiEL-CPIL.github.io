"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_poly = None
    best_nonzero_in_square = float('inf')
    
    def count_nonzero(coeff_list):
        """Count nonzero entries in coefficient list (ascending degree)."""
        return sum(1 for c in coeff_list if c != 0)
    
    def square_of_poly_fast(exponents, coeffs):
        """
        Compute exact square using numpy for speed.
        Returns (nonzero_count, max_degree)
        """
        n = len(exponents)
        if n == 0:
            return 0, 0
        
        # Use dictionary approach but optimized
        square_dict = defaultdict(int)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(i, n):
                ej = exponents[j]
                cj = coeffs[j]
                deg = ei + ej
                if i == j:
                    square_dict[deg] += ci * cj
                else:
                    square_dict[deg] += 2 * ci * cj
        
        if not square_dict:
            return 0, 0
        
        max_deg = max(square_dict.keys())
        nz = sum(1 for v in square_dict.values() if v != 0)
        return nz, max_deg
    
    def validate(poly_coeff_list, exactly_12_terms=True):
        """Validate the polynomial meets constraints."""
        if len(poly_coeff_list) > 41:
            return False
        if len(poly_coeff_list) == 0:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        nonzero = count_nonzero(poly_coeff_list)
        if exactly_12_terms and nonzero != 12:
            return False
        return True
    
    def evaluate_fast(exponents, coeffs):
        """Fast evaluation: return nonzero count in square."""
        nz, _ = square_of_poly_fast(exponents, coeffs)
        return nz
    
    def try_poly(poly_coeff_list):
        nonlocal best_poly, best_nonzero_in_square
        if not validate(poly_coeff_list, exactly_12_terms=True):
            return False
        exponents = [i for i, c in enumerate(poly_coeff_list) if c != 0]
        coeffs = [c for c in poly_coeff_list if c != 0]
        nz = evaluate_fast(exponents, coeffs)
        if nz < best_nonzero_in_square:
            best_nonzero_in_square = nz
            best_poly = list(poly_coeff_list)
            return True
        return False
    
    def try_exponents_coeffs(exponents, coeffs):
        """Try a given set of exponents and coefficients directly."""
        nonlocal best_poly, best_nonzero_in_square
        if len(exponents) != 12:
            return False
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return False
        if exponents[-1] >= 41:
            return False
        nz = evaluate_fast(exponents, coeffs)
        if nz < best_nonzero_in_square:
            best_nonzero_in_square = nz
            # Build full coefficient list
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly
            return True
        return False
    
    # Phase 1: Try structured patterns with arithmetic progressions
    # Key insight: if exponents are in AP and coeffs follow pattern, cancellations occur
    
    # Try AP exponents: {a, a+d, a+2d, ..., a+11d}
    for d in range(1, 8):  # step size
        for a in range(0, min(41 - 11*d, 10)):  # starting exponent
            if time.time() > deadline - 15:
                break
            exponents = [a + i*d for i in range(12)]
            if exponents[-1] >= 41:
                continue
            
            # Try various coefficient patterns
            patterns = [
                [1]*12,
                [1 if i%2==0 else -1 for i in range(12)],
                [1 if (i//2)%2==0 else -1 for i in range(12)],
                [1 if (i//3)%2==0 else -1 for i in range(12)],
                [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
                [1,-2,1,-2,1,-2,1,-2,1,-2,1,-2],
            ]
            for cp in patterns:
                if time.time() > deadline - 15:
                    break
                try_exponents_coeffs(exponents, cp)
    
    # Phase 2: Try symmetric exponent sets (palindromic)
    # Exponents: {c, c+d, c+2d, ..., c+5d, c+6d, ..., c+10d, c+11d} where c+5d is center
    # Or: symmetric around some center
    
    # Try: exponents centered around mid
    for total_span in range(11, 41):
        if time.time() > deadline - 15:
            break
        if total_span < 11:
            continue
        
        # Try symmetric placements
        # Place 12 terms symmetrically
        # e.g., {0, k, 2k, 3k, 4k, 5k, 6k, 7k, 8k, 9k, 10k, 11k} is AP
        # Try: {0, a, b, c, d, e, f, g, h, i, j, total_span} with symmetry
        
        # Simpler: try palindromic coefficient patterns with AP exponents
        exponents = list(range(0, total_span + 1, (total_span + 11) // 12)) if total_span >= 11 else None
        if exponents is None or len(exponents) < 12:
            continue
        exponents = exponents[:12]
        
        # Palindromic coeffs
        pal_patterns = [
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1],
            [1,2,3,4,5,6,6,5,4,3,2,1],
            [1,-1,1,-1,1,-1,-1,1,-1,1,-1,1],
        ]
        for cp in pal_patterns:
            if time.time() > deadline - 15:
                break
            if len(cp) == 12 and all(abs(c) <= 64 for c in cp):
                try_exponents_coeffs(exponents, cp)
    
    # Phase 3: Randomized search with smart initialization
    random.seed(42)
    
    def random_search(num_trials=50000):
        nonlocal best_poly, best_nonzero_in_square
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Strategy: choose degree and 12 exponents
            degree = random.randint(12, 40)
            
            # Choose 12 exponents - bias toward structured patterns
            r = random.random()
            if r < 0.3:
                # AP-like
                d = random.randint(1, 5)
                a = random.randint(0, max(0, 40 - 11*d))
                exponents = [a + i*d for i in range(12)]
                if exponents[-1] > degree:
                    continue
            elif r < 0.5:
                # Clustered
                base = random.randint(0, degree - 11)
                offsets = sorted(random.sample(range(0, 12), 11))
                exponents = [base + o for o in offsets]
                if exponents[-1] > degree:
                    continue
            else:
                # Random
                exponents = sorted(random.sample(range(degree+1), 12))
            
            # Choose coefficients
            coeff_style = random.random()
            if coeff_style < 0.4:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif coeff_style < 0.6:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif coeff_style < 0.8:
                # Patterned
                pattern = random.choice([
                    [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                    [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                    [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                ])
                coeffs = list(pattern)
            else:
                coeffs = [random.randint(-3, 3) for _ in range(12)]
                # Ensure no zeros
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 3) * random.choice([-1, 1])
            
            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_exponents_coeffs(exponents, coeffs)
    
    random_search(80000)
    
    # Phase 4: Local search with coefficient optimization
    if best_poly is not None and time.time() < deadline - 10:
        def local_search(poly, max_iter=5000):
            nonlocal best_poly, best_nonzero_in_square
            current = list(poly)
            current_exp = [i for i, c in enumerate(current) if c != 0]
            current_coef = [c for c in current if c != 0]
            current_nz = evaluate_fast(current_exp, current_coef)
            
            best_local_nz = current_nz
            best_local_exp = list(current_exp)
            best_local_coef = list(current_coef)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Random perturbation
                exp_copy = list(current_exp)
                coef_copy = list(current_coef)
                
                r = random.random()
                if r < 0.4:
                    # Change a coefficient
                    idx = random.randint(0, 11)
                    new_c = random.choice([-3,-2,-1,1,2,3,4,5,6])
                    if new_c != coef_copy[idx] and new_c != 0:
                        coef_copy[idx] = new_c
                elif r < 0.7:
                    # Move an exponent
                    idx = random.randint(0, 11)
                    shift = random.choice([-3,-2,-1,1,2,3])
                    new_e = exp_copy[idx] + shift
                    if new_e >= 0 and new_e < 41:
                        # Check collision
                        if new_e not in exp_copy:
                            exp_copy[idx] = new_e
                            exp_copy.sort()
                        else:
                            continue
                else:
                    # Swap two exponents
                    i, j = random.sample(range(12), 2)
                    exp_copy[i], exp_copy[j] = exp_copy[j], exp_copy[i]
                    exp_copy.sort()
                
                # Validate
                if len(set(exp_copy)) != 12:
                    continue
                if any(e < 0 or e >= 41 for e in exp_copy):
                    continue
                if any(c == 0 or abs(c) > 64 for c in coef_copy):
                    continue
                
                nz = evaluate_fast(exp_copy, coef_copy)
                if nz < best_local_nz:
                    best_local_nz = nz
                    best_local_exp = list(exp_copy)
                    best_local_coef = list(coef_copy)
                    current_exp = list(exp_copy)
                    current_coef = list(coef_copy)
                    
                    # Update global best
                    if nz < best_nonzero_in_square:
                        best_nonzero_in_square = nz
                        max_deg = max(exp_copy)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(exp_copy, coef_copy):
                            poly_new[e] = c
                        best_poly = poly_new
            
            return best_local_exp, best_local_coef, best_local_nz
        
        best_exp, best_coef, best_nz = local_search(best_poly, 8000)
    
    # Phase 5: Final targeted search if time remains
    if time.time() < deadline - 8:
        # Try to find even sparser squares with focused search
        # Focus on coefficient values that promote cancellation
        for d in range(1, 6):
            if time.time() > deadline - 5:
                break
            for a in range(0, min(41 - 11*d, 8)):
                if time.time() > deadline - 5:
                    break
                exponents = [a + i*d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
                
                # Try more coefficient combinations
                for c0 in [-2,-1,1,2]:
                    for c1 in [-2,-1,1,2]:
                        if time.time() > deadline - 5:
                            break
                        # Alternate pattern
                        coeffs = [c0 if i%2==0 else c1 for i in range(12)]
                        try_exponents_coeffs(exponents, coeffs)
    
    # If no valid polynomial found, return a default valid one
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    if not validate(best_poly, exactly_12_terms=True):
        best_poly = [1] * 12
    
    return best_poly

# EVOLVE-BLOCK-END
