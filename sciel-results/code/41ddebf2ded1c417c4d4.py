"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations, product
from collections import defaultdict

def _square_support(coeffs):
    """Compute number of nonzero terms in the square of a polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    # square via convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    return sum(1 for c in sq if c != 0)

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _trim(coeffs):
    """Remove trailing zeros, ensure nonzero final coefficient."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return coeffs

def _make_poly_from_terms(terms, max_deg=40):
    """terms: list of (exponent, coeff). Returns coeff list ascending degree."""
    coeffs = [0] * (max_deg + 1)
    for e, c in terms:
        if 0 <= e <= max_deg:
            coeffs[e] += c
    return coeffs

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    
    best = [1] * 12  # fallback: 12 terms, square has 12*12=144 entries but many cancel? No, (1+x+...+x^11)^2 has 23 nonzero terms
    best_score = _square_support(best)
    
    # Strategy: try structured polynomials
    # 1. Try (x^a - 1)(x^b - 1)(x^c - 1)(x^d - 1) type products
    # 2. Try sums of geometric progressions
    # 3. Random search with local improvement
    
    # First, try some known good structures
    
    # Try P = (1 + x^a + x^b + x^c) * (1 + x^d + x^e + x^f) * (1 + x^g)
    # That's 4*4*1 = 16 terms max, need exactly 12
    
    # Try P = (1 + x^a + x^b + x^c + x^d + x^e) * (1 + x^f + x^g + x^h)
    # 6 * 3 = 18 terms, too many
    
    # Try P = (1 + x^a + x^b + x^c + x^d) * (1 + x^e + x^f + x^g)
    # 5 * 3 = 15 terms
    
    # Try P = (1 + x^a + x^b + x^c) * (1 + x^d + x^e + x^f + x^g + x^h)
    # 4 * 6 = 24 terms
    
    # Let me try: P = (1 + x^a)(1 + x^b)(1 + x^c)(1 + x^d)(1 + x^e)(1 + x^f)
    # That's 2^6 = 64 terms, way too many
    
    # Better: P = sum of 12 monomials with carefully chosen exponents
    
    # Try: P = (1 + x^a + x^(2a) + x^(3a) + x^(4a) + x^(5a)) * (1 + x^b)
    # 6 * 2 = 12 terms!
    # Square: [(1+x^a+x^(2a)+x^(3a)+x^(4a)+x^(5a))^2] * (1+x^b)^2
    # = [6 + 5x^a + 4x^(2a) + 3x^(3a) + 2x^(4a) + x^(5a) + x^(6a) + x^(7a) + x^(8a) + x^(9a) + x^(10a)] * [1 + 2x^b + x^(2b)]
    # The first factor has 11 nonzero terms. Multiplying by (1+2x^b+x^(2b)) gives up to 33 terms,
    # but if b is chosen so that shifts align, cancellations may occur.
    
    # Actually let's just do a computational search.
    
    # Approach: enumerate small exponent sets and coefficient patterns
    
    # Try polynomials of form: c0*x^0 + c1*x^a + c2*x^b + ... + c11*x^k
    # with small exponents and coefficients in {-1, 0, 1} or {-2,-1,0,1,2}
    
    # Let's try a more systematic approach:
    # Generate candidate 12-term polynomials and evaluate
    
    candidates = []
    
    # Type 1: (1 + x^a + x^(2a) + x^(3a) + x^(4a) + x^(5a)) * (1 + x^b)
    for a in range(1, 8):
        for b in range(1, 8):
            if b == a or b == 2*a or b == 3*a or b == 4*a or b == 5*a:
                continue
            terms = [(i*a, 1) for i in range(6)] + [(i*a + b, 1) for i in range(6)]
            # Check if exactly 12 distinct exponents
            exps = [e for e, c in terms]
            if len(set(exps)) != 12:
                continue
            coeffs = _make_poly_from_terms(terms, max_deg=40)
            if _count_nonzero(coeffs) == 12:
                candidates.append(coeffs)
    
    # Type 2: (1 + x^a + x^(2a) + x^(3a) + x^(4a) + x^(5a)) * (1 - x^b)
    for a in range(1, 8):
        for b in range(1, 8):
            if b == a or b == 2*a or b == 3*a or b == 4*a or b == 5*a:
                continue
            terms = [(i*a, 1) for i in range(6)] + [(i*a + b, -1) for i in range(6)]
            exps = [e for e, c in terms]
            if len(set(exps)) != 12:
                continue
            coeffs = _make_poly_from_terms(terms, max_deg=40)
            if _count_nonzero(coeffs) == 12:
                candidates.append(coeffs)
    
    # Type 3: (1 + x^a + x^(2a) + x^(3a) + x^(4a) + x^(5a) + x^(6a)) * (1 + x^b) - x^(7a+b)
    # That's 7*2 - 1 = 13, minus one = 12. But need to check.
    
    # Type 4: Sum of 12 terms with exponents forming an arithmetic progression
    for start in range(0, 5):
        for step in range(1, 5):
            exps = [start + i*step for i in range(12)]
            if max(exps) > 40:
                continue
            # Try all-ones
            coeffs = [0] * 41
            for e in exps:
                coeffs[e] = 1
            candidates.append(coeffs)
            # Try alternating signs
            coeffs2 = [0] * 41
            for i, e in enumerate(exps):
                coeffs2[e] = 1 if i % 2 == 0 else -1
            candidates.append(coeffs2)
    
    # Type 5: Random-ish structured
    import random
    random.seed(42)
    
    for trial in range(2000):
        if time.time() > deadline - 10:
            break
        # Pick 12 random exponents in [0, 40]
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [0] * 41
        for e in exps:
            coeffs[e] = random.choice([-2, -1, 1, 2])
        candidates.append(coeffs)
    
    # Evaluate all candidates
    for coeffs in candidates:
        if time.time() > deadline - 5:
            break
        cnz = _count_nonzero(coeffs)
        if cnz != 12:
            continue
        # Check coefficient bounds
        if any(abs(c) > 64 for c in coeffs):
            continue
        # Check nonzero final coefficient
        trimmed = _trim(coeffs)
        if len(trimmed) > 41:
            continue
        score = _square_support(trimmed)
        if score < best_score:
            best_score = score
            best = trimmed
    
    # Local improvement: try flipping signs or adjusting coefficients
    for _ in range(100):
        if time.time() > deadline - 3:
            break
        improved = False
        best_trimmed = _trim(best)
        for i in range(len(best_trimmed)):
            if best_trimmed[i] == 0:
                continue
            old_val = best_trimmed[i]
            for new_val in [-2, -1, 0, 1, 2]:
                if new_val == old_val:
                    continue
                trial = list(best_trimmed)
                trial[i] = new_val
                if _count_nonzero(trial) != 12:
                    continue
                if any(abs(c) > 64 for c in trial):
                    continue
                trial_trimmed = _trim(trial)
                score = _square_support(trial_trimmed)
                if score < best_score:
                    best_score = score
                    best = trial_trimmed
                    improved = True
                    break
        if not improved:
            break
    
    return _trim(best)
# EVOLVE-BLOCK-END
