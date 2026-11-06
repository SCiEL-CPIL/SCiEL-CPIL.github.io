"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import defaultdict

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
    
    def poly_from_terms(terms):
        """Build coefficient list from list of (exponent, coeff) pairs."""
        if not terms:
            return [0]
        max_exp = max(e for e, c in terms)
        p = [0] * (max_exp + 1)
        for e, c in terms:
            p[e] = c
        return p
    
    def product_poly(a_terms, b_terms):
        """Compute product of two polynomials given as term lists."""
        result = defaultdict(int)
        for ae, ac in a_terms:
            for be, bc in b_terms:
                result[ae + be] += ac * bc
        # Convert to list of (exponent, coeff) for nonzero terms
        terms = [(e, c) for e, c in sorted(result.items()) if c != 0]
        return terms
    
    best_nz = float('inf')
    best_p = None
    
    def try_candidate(p):
        nonlocal best_nz, best_p
        if is_valid(p):
            sq = square_coeffs(p)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_p = p[:]
    
    # Strategy: Construct P = A * B where A and B are 6-term polynomials.
    # P will have at most 36 terms from the product, but we need exactly 12.
    # We search over exponent structures for A and B, and coefficient patterns,
    # aiming for P to have exactly 12 nonzero terms with a square that has few nonzero terms.
    
    # Key insight: If A and B have "complementary" exponent structures,
    # the Minkowski sum A_exp + B_exp can have exactly 12 distinct values
    # where the coefficients don't cancel to zero.
    
    # Phase 1: Try structured exponent patterns for A and B
    # A uses exponents from a small set, B uses exponents offset from A's range
    
    # Try: A has 6 consecutive exponents starting at 0, B has 6 consecutive exponents starting at offset
    # This gives P exponents from offset to offset+10 (11 terms) or with some structure
    
    # Better: Use arithmetic progressions
    # A = sum_{i=0}^{5} a_i * x^{d1*i}, B = sum_{j=0}^{5} b_j * x^{d2*j + offset}
    # P exponents: {d1*i + d2*j + offset : i,j in 0..5}
    
    # For the square to be sparse, we want P to have a structure like:
    # P = c * (x^m + x^n + ...) where the cross terms in P^2 cancel
    
    # Let's try specific constructions:
    
    # Construction 1: A = 1 + x + x^2 + x^3 + x^4 + x^5, B = x^k - x^{k+1} + x^{k+2} - x^{k+3} + x^{k+4} - x^{k+5}
    # P = A * B. Let's compute for various k.
    
    for k in range(1, 30):
        a_terms = [(i, 1) for i in range(6)]
        b_terms = [(k + j, 1 if j % 2 == 0 else -1) for j in range(6)]
        p_terms = product_poly(a_terms, b_terms)
        if len(p_terms) == 12:
            p = poly_from_terms(p_terms)
            try_candidate(p)
    
    # Construction 2: A = 1 - x + x^2 - x^3 + x^4 - x^5, B = x^k + x^{k+1} + x^{k+2} + x^{k+3} + x^{k+4} + x^{k+5}
    for k in range(1, 30):
        a_terms = [(i, 1 if i % 2 == 0 else -1) for i in range(6)]
        b_terms = [(k + j, 1) for j in range(6)]
        p_terms = product_poly(a_terms, b_terms)
        if len(p_terms) == 12:
            p = poly_from_terms(p_terms)
            try_candidate(p)
    
    # Construction 3: Both A and B alternating signs
    for k in range(1, 30):
        a_terms = [(i, 1 if i % 2 == 0 else -1) for i in range(6)]
        b_terms = [(k + j, 1 if j % 2 == 0 else -1) for j in range(6)]
        p_terms = product_poly(a_terms, b_terms)
        if len(p_terms) == 12:
            p = poly_from_terms(p_terms)
            try_candidate(p)
    
    # Construction 4: A with step 1, B with step 2 (or vice versa)
    # A = 1 + x + x^2 + x^3 + x^4 + x^5, B = x^k + x^{k+2} + x^{k+4} + x^{k+6} + x^{k+8} + x^{k+10}
    for k in range(0, 20):
        a_terms = [(i, 1) for i in range(6)]
        b_terms = [(k + 2*j, 1) for j in range(6)]
        p_terms = product_poly(a_terms, b_terms)
        if len(p_terms) == 12:
            p = poly_from_terms(p_terms)
            try_candidate(p)
    
    # Construction 5: A with step 2, B with step 1
    for k in range(0, 20):
        a_terms = [(2*i, 1) for i in range(6)]
        b_terms = [(k + j, 1) for j in range(6)]
        p_terms = product_poly(a_terms, b_terms)
        if len(p_terms) == 12:
            p = poly_from_terms(p_terms)
            try_candidate(p)
    
    # Construction 6: Use coefficients with magnitude 2 to create more cancellation opportunities
    # A = 1 + 2x + x^2 + ... patterns
    for k in range(1, 25):
        for a_mag in [1, 2]:
            for b_mag in [1, 2]:
                a_terms = [(i, a_mag * (1 if i % 2 == 0 else -1)) for i in range(6)]
                b_terms = [(k + j, b_mag * (1 if j % 2 == 0 else -1)) for j in range(6)]
                p_terms = product_poly(a_terms, b_terms)
                if len(p_terms) == 12:
                    p = poly_from_terms(p_terms)
                    try_candidate(p)
    
    # Phase 2: Broader search over exponent structures
    # Try A with exponents {0, a, 2a, 3a, 4a, 5a} and B with exponents {k, k+b, k+2b, k+3b, k+4b, k+5b}
    for a in range(1, 4):
        for b in range(1, 4):
            for k in range(0, 30):
                a_terms = [(a*i, 1) for i in range(6)]
                b_terms = [(k + b*j, 1) for j in range(6)]
                p_terms = product_poly(a_terms, b_terms)
                if len(p_terms) == 12:
                    p = poly_from_terms(p_terms)
                    try_candidate(p)
                
                # Try with alternating signs
                a_terms = [(a*i, 1 if i % 2 == 0 else -1) for i in range(6)]
                b_terms = [(k + b*j, 1 if j % 2 == 0 else -1) for j in range(6)]
                p_terms = product_poly(a_terms, b_terms)
                if len(p_terms) == 12:
                    p = poly_from_terms(p_terms)
                    try_candidate(p)
    
    # Phase 3: Try specific known-good structures
    # P = (1 + x + x^2)(1 + x^3 + x^6)(1 + x^9) type constructions won't give 12 terms easily
    
    # Try: P with exponents forming two interleaved arithmetic progressions
    # e.g., exponents {0,2,4,6,8,10} and {1,3,5,7,9,11} with appropriate signs
    
    for sign_a in [1, -1]:
        for sign_b in [1, -1]:
            terms = []
            for i in range(6):
                terms.append((2*i, sign_a))
                terms.append((2*i + 1, sign_b))
            p = poly_from_terms(terms)
            try_candidate(p)
    
    # Try: P = sum_{i=0}^{11} c_i x^i where c_i follows a pattern that makes P^2 sparse
    # Use coefficients from {-2, -1, 0, 1, 2} but ensure exactly 12 nonzero
    
    # Phase 4: Direct search over 12-term polynomials with small coefficients
    # Try consecutive exponents 0..11 with coefficients in {-1, 0, 1} ensuring 12 nonzero
    # That means all 12 must be nonzero, so coefficients in {-1, 1}
    for mask in range(4096):
        coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        p = coeffs[:]
        try_candidate(p)
    
    # Try with coefficients in {-2, -1, 1, 2}
    # Sample a subset of patterns
    for _ in range(5000):
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        p = coeffs[:]
        try_candidate(p)
    
    # Try non-consecutive exponents
    # Sample random 12-exponent subsets within [0, 40]
    for _ in range(3000):
        exps = sorted(random.sample(range(41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        p = [0] * 41
        for e, c in zip(exps, coeffs):
            p[e] = c
        # Trim trailing zeros
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        try_candidate(p)
    
    # Phase 5: Try palindromic and anti-palindromic structures
    # P(x) = x^11 * P(1/x) (palindromic) or P(x) = -x^11 * P(1/x) (anti-palindromic)
    for mask in range(2048):  # 2^11 for first 11 coefficients
        first_half = [1 if (mask >> i) & 1 else -1 for i in range(11)]
        # Palindromic: c_i = c_{11-i}
        coeffs = first_half + [first_half[0]]
        p = coeffs[:]
        try_candidate(p)
        
        # Anti-palindromic: c_i = -c_{11-i}
        coeffs_anti = first_half + [-first_half[0]]
        p_anti = coeffs_anti[:]
        try_candidate(p_anti)
    
    # Phase 6: Try products with more varied coefficient patterns
    for k in range(0, 35):
        for a_sign in range(64):  # 2^6 sign patterns for A
            for b_sign in range(64):  # 2^6 sign patterns for B
                a_terms = [(i, 1 if (a_sign >> i) & 1 else -1) for i in range(6)]
                b_terms = [(k + j, 1 if (b_sign >> j) & 1 else -1) for j in range(6)]
                p_terms = product_poly(a_terms, b_terms)
                if len(p_terms) == 12:
                    p = poly_from_terms(p_terms)
                    try_candidate(p)
    
    # If no valid candidate found, fall back to seed
    if best_p is None:
        return [1]*12
    
    return best_p
# EVOLVE-BLOCK-END
