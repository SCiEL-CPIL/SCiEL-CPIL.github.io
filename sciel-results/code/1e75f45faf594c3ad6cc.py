"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    
    def square_coeffs(p):
        """Compute P^2 coefficients from P's coefficient list."""
        n = len(p)
        result = [0] * (2*n - 1)
        for i in range(n):
            for j in range(n):
                result[i+j] += p[i] * p[j]
        return result
    
    def count_nonzero(poly_coeffs):
        """Count nonzero terms in a polynomial given as coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def is_valid(p):
        """Check if p satisfies all constraints."""
        if len(p) > 41:
            return False
        if len(p) == 0 or p[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in p):
            return False
        if sum(1 for c in p if c != 0) != 12:
            return False
        return True
    
    best_nz = float('inf')
    best_p = None
    
    # Strategy: Try various constructions and search for good candidates
    # Key insight: use exponent patterns that create many collisions in E+E,
    # and coefficient patterns that cause cancellations.
    
    # Approach 1: Consecutive exponents with optimized coefficients
    # P = sum c_i x^i for i in some set of 12 exponents
    # Try consecutive exponents with various coefficient patterns
    
    # Try all subsets of coefficient patterns for consecutive exponents 0..11
    # with coefficients from {-2,-1,1,2}
    
    # First, try some hand-crafted candidates
    candidates = []
    
    # Consecutive exponents 0..11 with various coefficient patterns
    base_exponents = list(range(12))
    
    # Try patterns with coefficients in {-1, 1}
    for mask in range(4096):  # 2^12 patterns
        coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        p = coeffs[:]
        if is_valid(p):
            sq = square_coeffs(p)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_p = p[:]
    
    # Try with coefficients in {-2, -1, 1, 2} for some positions
    # Focus on patterns that might create cancellations
    # Try: first 6 have one sign, last 6 have opposite
    for sign_pattern in [
        [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
        [1,-1,-1,1,1,1,-1,1,1,-1,-1,-1],
    ]:
        p = sign_pattern[:]
        if is_valid(p):
            sq = square_coeffs(p)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_p = p[:]
    
    # Try with some coefficients of magnitude 2
    # P with coefficients [1,1,1,1,1,1,2,2,2,2,2,2]
    for mag_pattern in [
        [1,1,1,1,1,1,2,2,2,2,2,2],
        [1,1,1,1,1,1,-2,-2,-2,-2,-2,-2],
        [2,2,2,2,2,2,1,1,1,1,1,1],
        [2,2,2,2,2,2,-1,-1,-1,-1,-1,-1],
    ]:
        p = mag_pattern[:]
        if is_valid(p):
            sq = square_coeffs(p)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_p = p[:]
    
    # Try non-consecutive exponents to create more structure
    # Exponents: 0,1,2,3,4,5,6,7,8,9,10,12 (skip 11)
    exp_sets = [
        list(range(12)),
        [0,1,2,3,4,5,6,7,8,9,10,12],
        [0,1,2,3,4,5,6,7,8,9,11,12],
        [0,1,2,3,4,5,6,7,8,10,11,12],
        [0,1,2,3,4,5,6,7,9,10,11,12],
        [0,1,2,3,4,5,6,8,9,10,11,12],
        [0,1,2,3,4,5,7,8,9,10,11,12],
        [0,1,2,3,4,6,7,8,9,10,11,12],
        [0,1,2,3,5,6,7,8,9,10,11,12],
        [0,1,2,4,5,6,7,8,9,10,11,12],
        [0,1,3,4,5,6,7,8,9,10,11,12],
        [0,2,3,4,5,6,7,8,9,10,11,12],
    ]
    
    for exps in exp_sets:
        if len(exps) != 12:
            continue
        # Try all +/-1 coefficient patterns
        for mask in range(4096):
            coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            # Build polynomial
            max_exp = max(exps)
            p = [0] * (max_exp + 1)
            for i, e in enumerate(exps):
                p[e] = coeffs[i]
            if is_valid(p):
                sq = square_coeffs(p)
                nz = count_nonzero(sq)
                if nz < best_nz:
                    best_nz = nz
                    best_p = p[:]
    
    # Try with gaps: exponents like 0,1,2,3,4,5,10,11,12,13,14,15
    gap_exp_sets = [
        [0,1,2,3,4,5,10,11,12,13,14,15],
        [0,1,2,3,4,5,9,10,11,12,13,14],
        [0,1,2,3,4,5,8,9,10,11,12,13],
        [0,1,2,3,4,5,7,8,9,10,11,12],
        [0,2,4,6,8,10,12,14,16,18,20,22],
        [0,2,4,6,8,10,11,13,15,17,19,21],
    ]
    
    for exps in gap_exp_sets:
        if len(exps) != 12:
            continue
        for mask in range(4096):
            coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            max_exp = max(exps)
            if max_exp >= 41:
                continue
            p = [0] * (max_exp + 1)
            for i, e in enumerate(exps):
                p[e] = coeffs[i]
            if is_valid(p):
                sq = square_coeffs(p)
                nz = count_nonzero(sq)
                if nz < best_nz:
                    best_nz = nz
                    best_p = p[:]
    
    # Try product construction: P = A * B where A and B are 6-term polynomials
    # A has exponents from set SA, B has exponents from set SB
    # P's exponents are SA + SB (Minkowski sum)
    # We want |{a+b : a in SA, b in SB} with nonzero coeff| = 12
    
    # Try simple cases
    # A = x^0 + x^1 + x^2 + x^3 + x^4 + x^5, B = x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    # P = A*B has exponents 6..16, all with coefficient 1 (no cancellation)
    # That's 11 terms, not 12.
    
    # A = x^0 + x^1 + x^2 + x^3 + x^4 + x^5, B = x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    # We computed this gives 6 terms.
    
    # Let's try: A = 1+x+x^2+x^3+x^4+x^5, B = x^6+x^8+x^10+x^12+x^14+x^16
    # P exponents: 6,7,8,9,10,11, 8,9,10,11,12,13, 10,11,12,13,14,15, 12,13,14,15,16,17, 14,15,16,17,18,19, 16,17,18,19,20,21
    # Many collisions. Let me just compute.
    
    product_candidates = [
        # (A_exponents, A_coeffs, B_exponents, B_coeffs)
        ([0,1,2,3,4,5], [1,1,1,1,1,1], [6,7,8,9,10,11], [1,1,1,1,1,1]),
        ([0,1,2,3,4,5], [1,1,1,1,1,1], [6,7,8,9,10,11], [1,-1,1,-1,1,-1]),
        ([0,1,2,3,4,5], [1,1,1,1,1,1], [6,8,10,12,14,16], [1,1,1,1,1,1]),
        ([0,1,2,3,4,5], [1,1,1,1,1,1], [6,8,10,12,14,16], [1,-1,1,-1,1,-1]),
        ([0,1,2,3,4,5], [1,-1,1,-1,1,-1], [6,7,8,9,10,11], [1,1,1,1,1,1]),
        ([0,1,2,3,4,5], [1,-1,1,-1,1,-1], [6,7,8,9,10,11], [1,-1,1,-1,1,-1]),
        ([0,2,4,6,8,10], [1,1,1,1,1,1], [0,1,2,3,4,5], [1,1,1,1,1,1]),
        ([0,2,4,6,8,10], [1,1,1,1,1,1], [0,1,2,3,4,5], [1,-1,1,-1,1,-1]),
    ]
    
    for a_exps, a_coeffs, b_exps, b_coeffs in product_candidates:
        # Compute P = A * B
        max_exp = max(max(a_exps) + max(b_exps), 0)
        p = [0] * (max_exp + 1)
        for ae, ac in zip(a_exps, a_coeffs):
            for be, bc in zip(b_exps, b_coeffs):
                p[ae + be] += ac * bc
        if is_valid(p):
            sq = square_coeffs(p)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_p = p[:]
    
    # If no valid candidate found, fall back to seed
    if best_p is None:
        return [1]*12
    
    return best_p
# EVOLVE-BLOCK-END
