"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    # Helper: compute sparse square coefficient count
    def square_nonzero_count(coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(coeffs)
        if n == 0:
            return 0
        # Build dict of exponent -> coefficient for the square
        sq = {}
        for i in range(n):
            ei, ci = coeffs[i]
            for j in range(i, n):
                ej, cj = coeffs[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)
    
    # Helper: validate and trim
    def validate_and_trim(poly):
        """Ensure poly meets constraints, return trimmed list or None."""
        if not poly:
            return None
        # Trim trailing zeros
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        if not poly:
            return None
        if len(poly) > 41:
            return None
        if poly[-1] == 0:
            return None
        nonzero_count = sum(1 for c in poly if c != 0)
        if nonzero_count != 12:
            return None
        if any(abs(c) > 64 for c in poly):
            return None
        return poly
    
    # Strategy 1: Try structured exponent patterns with small coefficients
    # Exponents should be within [0, 40]
    
    # Try polynomials with exponents in arithmetic progression
    # P(x) = sum of c_i * x^(a + i*d) for i in 0..11
    # Try various (a, d, coefficients)
    
    # First, try simple patterns: coefficients in {-1, 0, 1} with 12 ones
    # Exponents: try various sets
    
    # Precompute some candidate exponent sets
    # We want pair-sums to collide as much as possible
    
    # Arithmetic progression: exponents = a, a+d, a+2d, ..., a+11d
    # Need a + 11d <= 40
    # For d=1: a <= 29
    # For d=2: a <= 18
    # etc.
    
    # Try d=1, a=0: exponents 0,1,...,11
    # Coefficients: try all sign patterns? 2^12 = 4096, feasible
    # But also need to try different coefficient magnitudes
    
    # Let's be systematic:
    # Phase 1: Try AP exponents with coefficients in {-1, 1}
    # Phase 2: Try AP exponents with coefficients in {-2,-1,1,2}
    # Phase 3: Try other exponent structures
    
    # Phase 1: AP with ±1 coefficients
    for d in range(1, 4):  # d = 1, 2, 3
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            # Try all 2^12 sign patterns
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    # Return a valid fallback
                    return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                coeffs = []
                for i in range(12):
                    c = 1 if (signs_mask >> i) & 1 else -1
                    coeffs.append((exponents[i], c))
                score = square_nonzero_count(coeffs)
                if score < best_score:
                    best_score = score
                    # Build polynomial
                    poly = [0] * (exponents[-1] + 1)
                    for exp, c in coeffs:
                        poly[exp] = c
                    best_poly = validate_and_trim(poly)
                    if best_score <= 2:
                        # Can't do better than 2 (1 is impossible for 12 terms)
                        # Actually 1 is possible if it's a perfect square, but 12 terms can't be a square of fewer terms easily
                        # Let's keep searching a bit more
                        pass
    
    # Phase 2: Try some random exponent sets
    random.seed(42)
    for trial in range(2000):
        if time.time() > deadline:
            break
        # Generate 12 random distinct exponents in [0, 40]
        exponents = sorted(random.sample(range(0, 41), 12))
        # Try a few coefficient patterns
        for coeff_pattern in [
            [1]*12,
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
            [1,1,1,1,1,1,1,1,1,1,1,-1],
            [1,1,1,1,1,-1,1,1,1,1,1,1],
        ]:
            coeffs = [(exponents[i], coeff_pattern[i]) for i in range(12)]
            score = square_nonzero_count(coeffs)
            if score < best_score:
                best_score = score
                poly = [0] * (exponents[-1] + 1)
                for exp, c in coeffs:
                    poly[exp] = c
                best_poly = validate_and_trim(poly)
    
    # Phase 3: Try palindromic structures
    # P(x) = x^k * Q(x) where Q is palindromic of degree 2m
    # For 12 terms, Q could have 12 terms palindromic
    # Palindromic: c_i = c_{d-i}
    # 12 terms: 6 pairs + possibly middle terms
    # Let's try Q with exponents 0,1,...,11 with palindromic coefficients
    # c_0=c_11, c_1=c_10, ..., c_5=c_6
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        # Base exponents 0..11, shifted by d_shift
        base_exps = list(range(12))
        shifted_exps = [e + d_shift for e in base_exps]
        if shifted_exps[-1] > 40:
            continue
        # Palindromic: coeffs[i] = coeffs[11-i]
        # 6 independent coefficients, each in {-2,-1,1,2}
        for c0 in [-2,-1,1,2]:
            for c1 in [-2,-1,1,2]:
                for c2 in [-2,-1,1,2]:
                    for c3 in [-2,-1,1,2]:
                        for c4 in [-2,-1,1,2]:
                            for c5 in [-2,-1,1,2]:
                                coeffs_list = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                # Check exactly 12 nonzero
                                if sum(1 for c in coeffs_list if c != 0) != 12:
                                    continue
                                coeffs = [(shifted_exps[i], coeffs_list[i]) for i in range(12)]
                                score = square_nonzero_count(coeffs)
                                if score < best_score:
                                    best_score = score
                                    poly = [0] * (shifted_exps[-1] + 1)
                                    for exp, c in coeffs:
                                        poly[exp] = c
                                    best_poly = validate_and_trim(poly)
    
    # Return best found, or a valid fallback
    if best_poly is not None:
        return best_poly
    
    # Fallback: 12 x^0 + ... just make a valid 12-term polynomial
    # x^0 + x^1 + x^2 + ... + x^11
    return [1]*12

# EVOLVE-BLOCK-END
