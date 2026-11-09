"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

    # We restrict to arithmetic progression supports 0..11.
    # P(x) = sum_{i=0}^{11} a_i x^i
    # We want to minimize the number of non-zero coefficients in P(x)^2.
    
    # Strategy:
    # 1. Use Linear Algebra over a finite field to find a sparse solution.
    #    Let P(x)^2 = Q(x). We want Q(x) to have many zero coefficients.
    #    The coefficients of Q are quadratic forms in a_i.
    #    However, if we work over F_2, 2*a_i*a_j = 0, so Q_k = sum_{i+j=k} a_i a_j (mod 2).
    #    This is a system of quadratic equations.
    #    
    #    Alternative: Use the fact that if P(x) is a product of linear factors (x-r_i),
    #    the roots might be chosen to create cancellations.
    #    
    #    Better Approach: Direct Search for "Golay Polynomials" or similar structures.
    #    Known result: For length 12, can we get a square with very few terms?
    #    
    #    Let's try a hybrid:
    #    1. Random search with strong bias towards small integers {-1, 0, 1} and {-2, -1, 0, 1, 2}.
    #    2. Local search using gradient-like steps (try changing one coefficient to minimize count).
    #    3. Specifically, look for patterns like 1, -1, 1, -1... or block structures.
    
    best_count = 23
    best_poly = None
    
    # Heuristic: Small integers are more likely to cancel exactly.
    # Let's focus search on coefficients in {-1, 1} first, then {-2, -1, 1, 2}, etc.
    
    # Phase 1: Exhaustive-ish search on binary coefficients {-1, 1}
    # 2^12 = 4096 possibilities. This is fast.
    for mask in range(1 << 12):
        if time.time() > deadline:
            break
        poly = []
        for i in range(12):
            if (mask >> i) & 1:
                poly.append(1)
            else:
                poly.append(-1)
        
        # Ensure not all same (symmetry) - actually symmetry doesn't matter for count
        c = count_nonzero_square(poly)
        if c < best_count:
            best_count = c
            best_poly = poly[:]
    
    # Phase 2: Local search from the best binary solution, allowing values in {-2, -1, 1, 2}
    if best_poly:
        current = best_poly[:]
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                for new_val in [-2, -1, 1, 2]:
                    if new_val == current[i]:
                        continue
                    old_val = current[i]
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
    
    # Phase 3: Random restarts with small integers {-3, -2, -1, 1, 2, 3}
    # If Phase 1/2 didn't find something very sparse (e.g., < 10), do this.
    if best_count > 10:
        for r in range(500):
            if time.time() > deadline:
                break
            # Random init in small range
            poly = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            c = count_nonzero_square(poly)
            if c < best_count:
                best_count = c
                best_poly = poly[:]
                if best_count <= 5:
                    break
            
            # Simple local search
            for step in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, 11)
                old_val = poly[i]
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
                if new_val == old_val:
                    continue
                poly[i] = new_val
                new_c = count_nonzero_square(poly)
                if new_c < c:
                    c = new_c
                    if c < best_count:
                        best_count = c
                        best_poly = poly[:]
                        if best_count <= 5:
                            break
                else:
                    poly[i] = old_val

    # Phase 4: If still not good, try specific structured patterns
    # e.g., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 -> Square: 1, 0.., 2, 0.., 1 (3 non-zeros)
    # Wait, the problem says "exactly 12 nonzero terms" in the INPUT polynomial.
    # So we cannot have zeros in the input.
    # My Phase 1 used {-1, 1}, which has 12 non-zero terms. Correct.
    
    # Let's verify the best_poly found so far
    if best_poly is None:
        # Fallback
        return [1] * 12
        
    # Ensure constraints:
    # 1. Length 12 (<= 41)
    # 2. Nonzero final coeff (last element of list is coeff of x^11)
    # 3. Entries in [-64, 64]
    # 4. Exactly 12 nonzero terms.
    
    # My search ensures 12 non-zero terms because I never pick 0 in the value sets used.
    # Values used: {-1, 1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}. All non-zero.
    
    return best_poly

# EVOLVE-BLOCK-END
