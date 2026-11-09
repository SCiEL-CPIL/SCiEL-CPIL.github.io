"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    def count_nonzero_square(coeffs):
        """Calculate the number of non-zero coefficients in the square of the polynomial."""
        # coeffs is a list of 12 integers
        # Square has degree 22
        square_coeffs = [0] * 23
        for i in range(12):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, 12):
                aj = coeffs[j]
                if aj == 0:
                    continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    # Strategy:
    # Represent P(x) = A(x) * B(x) where deg(A)=3, deg(B)=2 (or similar splits).
    # This ensures P has at most 4+3-1 = 6 terms if A,B are sparse, 
    # but we need EXACTLY 12 non-zero terms in P.
    # Wait, the constraint is "exactly 12 nonzero terms" in the INPUT polynomial.
    # If P = A*B, and A, B are dense (all coeffs non-zero), P will be dense (12 terms).
    # If A, B have zeros, P might have fewer than 12 terms.
    # We must ensure the resulting P has exactly 12 non-zero coefficients.
    
    # Let's use A of length 4 and B of length 3.
    # Coeffs of A: a0, a1, a2, a3
    # Coeffs of B: b0, b1, b2
    # P = A*B will have length 6? No, deg(A)=3, deg(B)=2 -> deg(P)=5. Length 6.
    # We need length 12. So deg(P)=11.
    # Let deg(A)=5, deg(B)=6? Or deg(A)=4, deg(B)=7?
    # Let's try deg(A)=3, deg(B)=8? No, B would be large.
    # Let's try deg(A)=5, deg(B)=5. P has deg 10 (length 11). We need length 12 (deg 11).
    # Let deg(A)=5, deg(B)=6. P has deg 11 (length 12).
    # A: a0..a5 (6 coeffs)
    # B: b0..b6 (7 coeffs)
    # Total variables: 13.
    # We need P to have exactly 12 non-zero coeffs.
    # If A and B are chosen such that no cancellation occurs in the product, P is dense.
    # We search for A, B with small integer coefficients such that:
    # 1. P = A*B has exactly 12 non-zero coefficients.
    # 2. P^2 has minimal non-zero coefficients.
    
    best_count = 23
    best_poly = None
    
    # Value set for coefficients
    vals = [-2, -1, 1, 2]
    
    # To speed up, we can fix one coefficient to 1 (scale invariance of sparsity count? 
    # No, scaling doesn't change zero pattern, but we need integer coeffs.
    # Actually, if we scale P by k, P^2 scales by k^2, zero pattern unchanged.
    # So we can fix a0=1 without loss of generality for the zero pattern, 
    # provided gcd of coeffs is 1? No, just fix a0=1 to break symmetry.
    
    # Iterate over possible A (length 6) and B (length 7)
    # This is 4^6 * 4^7 = 4^13 = 67 million. Too slow for Python loops.
    # We need to prune or use a smarter search.
    
    # Alternative: Randomized search with local improvement.
    # Initialize P directly? No, the directive asks for the decomposition strategy.
    # Let's use the decomposition to generate candidates.
    
    # We can reduce the search space by fixing the first coefficient of A to 1.
    # And maybe the last coefficient of B to 1?
    
    # Let's try a hybrid:
    # 1. Randomly generate A (len 6) and B (len 7) from small ints.
    # 2. Compute P = A*B.
    # 3. Check if P has exactly 12 non-zero coeffs.
    # 4. If yes, compute count_nonzero_square(P).
    # 5. Keep track of best.
    
    # To make this efficient, we can also do local search on the coefficients of A and B.
    
    # Phase 1: Random sampling of (A, B) pairs
    num_random_trials = 5000
    for _ in range(num_random_trials):
        if time.time() > deadline:
            break
        
        # Random A
        a = [random.choice(vals) for _ in range(6)]
        # Random B
        b = [random.choice(vals) for _ in range(7)]
        
        # Ensure non-zero leading coeffs? Not strictly necessary, but helps with length.
        # If a[5]==0 or b[6]==0, deg(P) < 11, so length < 12.
        # We need length exactly 12 (indices 0..11).
        # So a[5] != 0 and b[6] != 0.
        if a[5] == 0 or b[6] == 0:
            continue
            
        # Compute P = A * B
        # Convolution
        p = [0] * 12
        for i in range(6):
            if a[i] == 0: continue
            for j in range(7):
                if b[j] == 0: continue
                p[i+j] += a[i] * b[j]
                
        # Check if exactly 12 non-zero terms
        if sum(1 for x in p if x != 0) != 12:
            continue
            
        # Check bounds [-64, 64]
        if any(abs(x) > 64 for x in p):
            continue
            
        c = count_nonzero_square(p)
        if c < best_count:
            best_count = c
            best_poly = p[:]
            if best_count <= 5:
                break

    # Phase 2: Local search on the best (A, B) or P?
    # It's easier to do local search on P directly if we found a good one, 
    # but the decomposition constraint might be lost.
    # However, the task doesn't require P to be factorable, only that it's a valid polynomial.
    # So once we find a candidate P, we can refine P directly using local search,
    # ignoring the factorization constraint for the refinement step.
    
    if best_poly:
        current = best_poly[:]
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                # Try changing coefficient i to a nearby value
                # We must keep it non-zero (exactly 12 non-zero terms)
                # And within [-64, 64]
                old_val = current[i]
                # Try small perturbations
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val == 0:
                        continue
                    if abs(new_val) > 64:
                        continue
                    current[i] = new_val
                    # Check if still 12 non-zero (it is, since we didn't make it 0)
                    new_count = count_nonzero_square(current)
                    if new_count < current_count:
                        current_count = new_count
                        if current_count < best_count:
                            best_count = current_count
                            best_poly = current[:]
                        improved = True
                    else:
                        current[i] = old_val
                # Also try larger jumps?
                for new_val in [-3, -2, 2, 3]:
                    if new_val == old_val:
                        continue
                    if abs(new_val) > 64:
                        continue
                    current[i] = new_val
                    new_count = count_nonzero_square(current)
                    if new_count < current_count:
                        current_count = new_count
                        if current_count < best_count:
                            best_count = current_count
                            best_poly = current[:]
                        improved = True
                    else:
                        current[i] = old_val

    # Phase 3: If we haven't found a good one, fallback to the previous strategy's best known or random
    if best_poly is None:
        # Fallback: Simple random search on P directly
        for _ in range(1000):
            if time.time() > deadline:
                break
            p = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            if sum(1 for x in p if x != 0) != 12:
                continue
            if any(abs(x) > 64 for x in p):
                continue
            c = count_nonzero_square(p)
            if c < best_count:
                best_count = c
                best_poly = p[:]

    if best_poly is None:
        return [1] * 12

    return best_poly

# EVOLVE-BLOCK-END
