import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s budget, 10s safety margin
    
    best_nonzero = 144  # baseline: 12*12 with no cancellation
    best_poly = None
    
    def count_nonzero(poly_coeffs):
        """Count nonzero entries in a coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def compute_square_nonzero(coeffs):
        """Given coefficient list (ascending degree), compute square and count nonzero terms."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    sq[i + j] += val
                else:
                    sq[i + j] += 2 * val
        # Trim trailing zeros
        while len(sq) > 1 and sq[-1] == 0:
            sq.pop()
        return count_nonzero(sq)

    def make_poly(exponents, coefficients, max_deg):
        """Create coefficient list from exponents and coefficients."""
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] = c
        return coeffs

    def get_nonzero_indices(poly):
        return [i for i, c in enumerate(poly) if c != 0]

    def refine_coefficients(poly, max_iters=20, coeff_range=6):
        """
        Iteratively improve coefficients for fixed exponents.
        For each nonzero term, try changing its coefficient to a nearby integer
        and keep the change if it reduces the nonzero count of the square.
        """
        current_poly = poly[:]
        current_nz = compute_square_nonzero(current_poly)
        nz_indices = get_nonzero_indices(current_poly)
        
        for _ in range(max_iters):
            improved = False
            # Shuffle to avoid bias, but keep deterministic seed for reproducibility in this scope?
            # Using random for diversity
            random.shuffle(nz_indices)
            for idx in nz_indices:
                orig_val = current_poly[idx]
                best_val = orig_val
                best_nz = current_nz
                
                # Try values in range [orig_val - range, orig_val + range]
                for delta in range(-coeff_range, coeff_range + 1):
                    if delta == 0:
                        continue
                    new_val = orig_val + delta
                    if new_val == 0:
                        continue # Must remain nonzero
                    if abs(new_val) > 64:
                        continue
                    
                    current_poly[idx] = new_val
                    new_nz = compute_square_nonzero(current_poly)
                    
                    if new_nz < best_nz:
                        best_nz = new_nz
                        best_val = new_val
                        improved = True
                    elif new_nz == best_nz:
                        # Tie-breaking: prefer smaller absolute coefficient?
                        if abs(new_val) < abs(best_val):
                            best_val = new_val
                            improved = True # Slight improvement in "cost"
                
                current_poly[idx] = best_val
                current_nz = best_nz
            
            if not improved:
                break
            if time.time() > deadline - 10:
                break
        
        return current_poly, current_nz

    def refine_exponents(poly, max_iters=20):
        """
        Iteratively improve exponents for fixed coefficients.
        Try shifting one term to a nearby empty slot.
        """
        current_poly = poly[:]
        current_nz = compute_square_nonzero(current_poly)
        nz_indices = get_nonzero_indices(current_poly)
        
        for _ in range(max_iters):
            improved = False
            random.shuffle(nz_indices)
            for idx in nz_indices:
                orig_val = current_poly[idx]
                best_idx = idx
                best_nz = current_nz
                
                # Try shifting to nearby positions
                for shift in range(-3, 4):
                    if shift == 0:
                        continue
                    new_idx = idx + shift
                    if new_idx < 0 or new_idx >= len(current_poly):
                        # Extend if needed, but keep length <= 41
                        if new_idx >= 41:
                            continue
                        if new_idx < 0:
                            continue
                        
                    # Check if slot is empty
                    if current_poly[new_idx] != 0:
                        continue
                    
                    # Apply change
                    current_poly[idx] = 0
                    current_poly[new_idx] = orig_val
                    
                    new_nz = compute_square_nonzero(current_poly)
                    
                    if new_nz < best_nz:
                        best_nz = new_nz
                        best_idx = new_idx
                        improved = True
                    else:
                        # Revert
                        current_poly[new_idx] = 0
                        current_poly[idx] = orig_val
                
                if best_idx != idx:
                    # Commit the best move found for this idx
                    current_poly[idx] = 0
                    current_poly[best_idx] = orig_val
                    current_nz = best_nz
                    nz_indices = get_nonzero_indices(current_poly) # Update indices
            
            if not improved:
                break
            if time.time() > deadline - 10:
                break
        
        return current_poly, current_nz

    def full_refine(poly):
        """
        Alternate between refining coefficients and exponents.
        """
        current_poly = poly[:]
        current_nz = compute_square_nonzero(current_poly)
        
        for _ in range(5): # 5 rounds of alternating
            if time.time() > deadline - 10:
                break
            
            # Refine coefficients
            refined_poly, nz_c = refine_coefficients(current_poly)
            if nz_c < current_nz:
                current_poly = refined_poly
                current_nz = nz_c
            
            # Refine exponents
            refined_poly, nz_e = refine_exponents(current_poly)
            if nz_e < current_nz:
                current_poly = refined_poly
                current_nz = nz_e
            
            # Check for convergence
            if time.time() > deadline - 10:
                break
        
        return current_poly, current_nz

    # Strategy 1: Arithmetic progression exponents with small integer coefficients
    coeff_choices = [-2, -1, 1, 2]
    
    for d in range(1, 15):
        for a in range(0, 5):
            exponents = [a + i * d for i in range(12)]
            max_deg = max(exponents)
            
            # Try all-1 first
            coeffs = [1] * 12
            poly = make_poly(exponents, coeffs, max_deg)
            nz = compute_square_nonzero(poly)
            if nz < best_nonzero:
                best_nonzero = nz
                best_poly = poly[:]
            
            # Try alternating signs
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            poly = make_poly(exponents, coeffs, max_deg)
            nz = compute_square_nonzero(poly)
            if nz < best_nonzero:
                best_nonzero = nz
                best_poly = poly[:]
            
            # Try palindromic coefficients
            for c1 in coeff_choices:
                for c2 in coeff_choices:
                    for c3 in coeff_choices:
                        for c4 in coeff_choices:
                            for c5 in coeff_choices:
                                for c6 in coeff_choices:
                                    coeffs = [c1, c2, c3, c4, c5, c6, c6, c5, c4, c3, c2, c1]
                                    poly = make_poly(exponents, coeffs, max_deg)
                                    nz = compute_square_nonzero(poly)
                                    if nz < best_nonzero:
                                        best_nonzero = nz
                                        best_poly = poly[:]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Strategy 2: P = A + x^k * A, where A is 6-term
    for k in range(20, 60):
        for d in range(1, 10):
            exps_A = [0, d, 2*d, 3*d, 4*d, 5*d]
            if k <= 5 * d:
                continue
            
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    for c3 in [-2, -1, 1, 2]:
                        for c4 in [-2, -1, 1, 2]:
                            for c5 in [-2, -1, 1, 2]:
                                for c6 in [-2, -1, 1, 2]:
                                    coeffs_A = [c1, c2, c3, c4, c5, c6]
                                    max_deg = max(exps_A) + k + max(exps_A)
                                    poly = [0] * (max_deg + 1)
                                    for e, c in zip(exps_A, coeffs_A):
                                        poly[e] = c
                                        poly[e + k] = c
                                    
                                    nz_poly = sum(1 for x in poly if x != 0)
                                    if nz_poly != 12:
                                        continue
                                    
                                    nz_sq = compute_square_nonzero(poly)
                                    if nz_sq < best_nonzero:
                                        best_nonzero = nz_sq
                                        best_poly = poly[:]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Strategy 3: Clustered exponent patterns
    cluster_patterns = [
        lambda: list(range(0, 6)) + list(range(10, 16)),
        lambda: list(range(0, 4)) + list(range(8, 12)) + list(range(16, 20)),
        lambda: [0,1,2,3,10,11,12,13,20,21,22,23],
    ]
    
    for pattern_fn in cluster_patterns:
        exponents = pattern_fn()
        if len(exponents) != 12 or len(set(exponents)) != 12:
            continue
        max_deg = max(exponents)
        
        coeffs = [1] * 12
        poly = make_poly(exponents, coeffs, max_deg)
        nz = compute_square_nonzero(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = poly[:]
        
        coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
        poly = make_poly(exponents, coeffs, max_deg)
        nz = compute_square_nonzero(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = poly[:]
        
        if time.time() > deadline:
            break

    # Strategy 4: Random search with hill climbing (kept for diversity, but refined)
    random.seed(42)
    
    def random_poly():
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
        max_deg = max(exps)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            poly[e] = c
        return poly

    # Generate a few random starts and refine them
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        poly = random_poly()
        refined_poly, nz = full_refine(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = refined_poly[:]

    # If we have a best_poly, do one final intensive refinement on it
    if best_poly is not None and time.time() < deadline - 10:
        final_poly, final_nz = full_refine(best_poly)
        if final_nz < best_nonzero:
            best_nonzero = final_nz
            best_poly = final_poly[:]

    # Return best found
    if best_poly is not None:
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        if len(best_poly) > 41:
            best_poly = best_poly[:41]
            while len(best_poly) > 1 and best_poly[-1] == 0:
                best_poly.pop()
        if best_poly[-1] == 0:
            for i in range(len(best_poly)-1, -1, -1):
                if best_poly[i] != 0:
                    best_poly = best_poly[:i+1]
                    break
        best_poly = [max(-64, min(64, c)) for c in best_poly]
        nz = sum(1 for c in best_poly if c != 0)
        if nz == 12 and len(best_poly) <= 41 and best_poly[-1] != 0:
            return best_poly
    
    return [1] * 12
