"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    def count_nonzero_square(coeffs):
        """Calculate the number of non-zero coefficients in the square of the polynomial."""
        # coeffs is a list of integers
        # Square has degree 2*len(coeffs)-2
        n = len(coeffs)
        square_coeffs = [0] * (2 * n - 1)
        for i in range(n):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, n):
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

    def is_valid_poly(p):
        if len(p) != 12:
            return False
        if sum(1 for x in p if x != 0) != 12:
            return False
        if any(abs(x) > 64 for x in p):
            return False
        if p[-1] == 0:
            return False
        return True

    best_count = 23
    best_poly = None

    # Strategy:
    # We look for polynomials P(x) with 12 non-zero terms.
    # A strong candidate for "few non-zero terms in square" is a polynomial
    # that is close to a perfect square of a sparse polynomial, or has
    # high symmetry (palindromic/anti-palindromic) which can cause cancellations.
    
    # The current program uses random search. 
    # This program uses a structured search:
    # 1. Enumerate palindromic polynomials with coefficients in {1, -1} (or small ints).
    #    Palindromic polynomials P(x) = x^11 Q(1/x) have specific properties.
    #    If P is palindromic, P^2 is palindromic.
    #    More importantly, if P = A^2 for some A, the square is A^4.
    #    But we need 12 terms.
    
    # Let's try a different structural approach:
    # Search for P(x) = sum_{i=0}^{11} c_i x^i where c_i in {1, -1}.
    # This is a "Littlewood" type polynomial.
    # There are 2^12 = 4096 such polynomials.
    # We can enumerate all of them, check validity, and compute square sparsity.
    # This is extremely fast.
    
    # Then, we can expand the search to coefficients in {1, -1, 2, -2} etc.
    # But first, let's exhaust the {1, -1} case.
    
    # Phase 1: Exhaustive search for coefficients in {1, -1}
    # We must ensure exactly 12 non-zero terms. Since all are 1 or -1, this is always true.
    # We must ensure non-zero final coefficient (always true).
    # We must ensure length 12 (always true).
    
    # Optimization: We can fix c_0 = 1 due to symmetry P(x) ~ -P(x) and P(x) ~ P(-x)?
    # Actually, P(x)^2 is invariant under P -> -P.
    # Also, if P(x) is a solution, P(-x) is a solution with same square sparsity.
    # So we can fix c_0 = 1.
    
    # There are 2^11 = 2048 candidates.
    
    for bits in range(2048):
        if time.time() > deadline:
            break
        
        # Construct polynomial
        p = [1] * 12
        for i in range(1, 12):
            if (bits >> (i - 1)) & 1:
                p[i] = -1
        
        # Check validity (redundant for this set, but good practice)
        if not is_valid_poly(p):
            continue
            
        c = count_nonzero_square(p)
        if c < best_count:
            best_count = c
            best_poly = p[:]
            if best_count <= 5:
                break

    # Phase 2: Local search from the best {1, -1} candidate
    # Expand coefficients to {1, -1, 2, -2, 3, -3} etc.
    # But we must keep exactly 12 non-zero terms.
    
    if best_poly is None:
        # Fallback if something went wrong (shouldn't happen)
        best_poly = [1] * 12
        best_count = count_nonzero_square(best_poly)

    current = best_poly[:]
    current_count = best_count

    # Local search: Try changing one coefficient at a time to a nearby value.
    # Values to try: current +/- 1, +/- 2, and some random jumps.
    # Constraint: new value != 0, abs(new value) <= 64.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = current[i]
            # Try a set of candidate values
            candidates = set()
            # Neighbors
            for d in range(-3, 4):
                if d == 0: continue
                nv = old_val + d
                if nv != 0 and abs(nv) <= 64:
                    candidates.add(nv)
            # Random jumps
            for _ in range(2):
                nv = random.randint(-64, 64)
                if nv != 0:
                    candidates.add(nv)
            
            for nv in candidates:
                if nv == old_val:
                    continue
                current[i] = nv
                new_count = count_nonzero_square(current)
                if new_count < current_count:
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_poly = current[:]
                    improved = True
                else:
                    current[i] = old_val

    # Phase 3: If we have time, try a different structural class:
    # Polynomials with sparse support but integer coefficients.
    # E.g., P(x) = x^11 + ... + 1 with some zeros? No, must have 12 non-zeros.
    # So support is fixed to 0..11 if we want length 12.
    # Wait, "length <= 41" and "exactly 12 nonzero terms".
    # The current assumption is that the polynomial is dense (indices 0..11).
    # But the support could be sparse! E.g., indices 0, 1, 3, 5, ...
    # If the support is sparse, the square might be sparser.
    
    # Let's check if a sparse support yields a better result.
    # If P(x) = sum c_i x^{e_i} with e_i distinct.
    # The square has terms x^{e_i + e_j}.
    # If the exponents are chosen such that e_i + e_j are distinct for most pairs,
    # the square will have ~12^2/2 = 72 terms.
    # If we choose exponents such that many collisions occur (e_i + e_j = e_k + e_l),
    # the square will be sparser.
    
    # The dense support 0..11 has many collisions (arithmetic progressions).
    # A geometric progression 1, 2, 4, 8... has NO collisions in sums?
    # 1+2=3, 1+4=5, 2+4=6, 1+8=9... All sums of distinct powers of 2 are distinct.
    # So sparse support with geometric exponents gives ~72 non-zero terms.
    # Dense support 0..11 gives ~23 terms (max degree 22).
    # So dense support is better for minimizing terms, provided coefficients cancel.
    
    # Therefore, sticking to dense support 0..11 is the right strategy.
    # The local search in Phase 2 explores the coefficient space.
    
    return best_poly

# EVOLVE-BLOCK-END
