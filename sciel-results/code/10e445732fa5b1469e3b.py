import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    
    best_poly = None
    best_nz_in_square = float('inf')
    
    def count_nonzero(coeff_list):
        return sum(1 for c in coeff_list if c != 0)

    def square_nonzero_count(exponents, coeffs):
        """
        Computes the number of non-zero terms in the square of the polynomial
        defined by exponents and coeffs.
        """
        n = len(exponents)
        if n == 0:
            return 0
            
        square_dict = defaultdict(int)
        # Optimization: sort by exponent to potentially break early or just iterate
        # We need all pairs (i, j) with i <= j
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
        
        nz = sum(1 for v in square_dict.values() if v != 0)
        return nz

    def validate_and_update(exponents, coeffs):
        nonlocal best_poly, best_nz_in_square
        
        # Basic validation
        if len(exponents) != 12:
            return False
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return False
        if any(e < 0 or e > 40 for e in exponents):
            return False
        if len(set(exponents)) != 12:
            return False
            
        nz = square_nonzero_count(exponents, coeffs)
        
        if nz < best_nz_in_square:
            best_nz_in_square = nz
            # Construct the full coefficient list
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            # Ensure length <= 41 and nonzero final coefficient
            if len(poly) <= 41 and poly[-1] != 0:
                best_poly = poly
            return True
        return False

    # Phase 1: Structured Initial Candidates
    # Try AP exponents with various coefficient patterns
    for d in range(1, 8):
        for a in range(0, 41 - 11*d):
            exponents = [a + i*d for i in range(12)]
            # Coefficient patterns
            patterns = [
                [1]*12,
                [1 if i%2==0 else -1 for i in range(12)],
                [1 if (i//2)%2==0 else -1 for i in range(12)],
                [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
                [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
            ]
            for cp in patterns:
                validate_and_update(exponents, cp)
                
            # Try some random coefficients for these exponents
            for _ in range(20):
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
                validate_and_update(exponents, coeffs)
                
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break

    # Phase 2: Local Search
    # Start from the best found so far, or a random valid one if none found
    if best_poly is None:
        # Fallback to a simple valid poly
        exponents = list(range(12))
        coeffs = [1]*12
        validate_and_update(exponents, coeffs)
        if best_poly is None:
             return [1]*12 # Should not happen

    # Extract current best exponents and coeffs
    current_exp = [i for i, c in enumerate(best_poly) if c != 0]
    current_coef = [c for c in best_poly if c != 0]
    
    # Local Search Loop
    # We will try to improve the current best by perturbing exponents and coefficients.
    # Use a simple hill-climbing with restarts or just long run.
    # Given the small search space (12 exponents in 0..40, coeffs in -64..64),
    # random perturbation is effective.
    
    random.seed(42)
    
    current_exp = list(current_exp)
    current_coef = list(current_coef)
    current_nz = best_nz_in_square
    
    for iteration in range(50000): # High number, break on time
        if time.time() > deadline - 5:
            break
        
        # Generate a neighbor
        exp_copy = list(current_exp)
        coef_copy = list(current_coef)
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Change an exponent
            idx = random.randint(0, 11)
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = exp_copy[idx] + shift
            if 0 <= new_e <= 40:
                if new_e not in exp_copy:
                    exp_copy[idx] = new_e
                    exp_copy.sort()
                else:
                    continue # Collision, skip
            else:
                continue
        elif move_type < 0.8:
            # Change a coefficient sign
            idx = random.randint(0, 11)
            coef_copy[idx] = -coef_copy[idx]
        else:
            # Change a coefficient value
            idx = random.randint(0, 11)
            new_c = random.choice([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
            if new_c != 0:
                coef_copy[idx] = new_c
        
        # Validate
        if len(set(exp_copy)) != 12:
            continue
        if any(abs(c) > 64 or c == 0 for c in coef_copy):
            continue
        if any(e < 0 or e > 40 for e in exp_copy):
            continue
            
        nz = square_nonzero_count(exp_copy, coef_copy)
        
        if nz < current_nz:
            current_nz = nz
            current_exp = exp_copy
            current_coef = coef_copy
            # Update global best
            validate_and_update(current_exp, current_coef)
        elif random.random() < 0.01:
            # Occasional restart or random jump to escape local minima
            # Just accept a random move sometimes? Or restart from best global?
            # Let's just keep going.
            pass

    # Final check: Ensure we return a valid polynomial
    if best_poly is None:
        return [1]*12
        
    # Ensure the returned list meets all constraints
    # Length <= 41
    if len(best_poly) > 41:
        # Truncate? No, max degree is bounded by exponents <= 40, so len <= 41.
        pass
        
    # Nonzero final coefficient
    if best_poly[-1] == 0:
        # Find last non-zero
        for i in range(len(best_poly)-1, -1, -1):
            if best_poly[i] != 0:
                best_poly = best_poly[:i+1]
                break
        if not best_poly:
            return [1]
            
    # Exactly 12 nonzero terms
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        # This should not happen if our update logic is correct, but safety check
        # If it fails, return a known valid one
        return [1]*12
        
    return best_poly
