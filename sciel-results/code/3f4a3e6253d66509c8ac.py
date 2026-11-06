"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _square_poly(coeffs):
    n = len(coeffs)
    result = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            result[i + j] += coeffs[i] * coeffs[j]
    # Trim trailing zeros
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    return result

def _valid_candidate(coeffs):
    if len(coeffs) > 41:
        return False
    if not coeffs or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if _count_nonzero(coeffs) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second safety margin
    
    best_coeffs = [1] * 12
    best_square_nonzero = _count_nonzero(_square_poly(best_coeffs))
    
    # Try the simple consecutive polynomial as baseline
    # [1,1,1,...,1] (12 ones) -> square has 23 nonzero terms
    
    # Strategy: try various structures
    # 1. Arithmetic progressions of exponents
    # 2. Try random sparse polynomials
    
    random.seed(42)
    
    # Try different exponent patterns
    for trial in range(500000):
        if time.time() > deadline:
            break
        
        # Generate random 12-term polynomial
        # Choose 12 distinct exponents from 0 to 40
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coeffs = [0] * (exponents[-1] + 1)
        for e in exponents:
            coeffs[e] = random.choice([-2, -1, 1, 2, 3, -3, 4, -4])
        
        if not _valid_candidate(coeffs):
            continue
        
        sq = _square_poly(coeffs)
        sq_nonzero = _count_nonzero(sq)
        
        if sq_nonzero < best_square_nonzero:
            best_square_nonzero = sq_nonzero
            best_coeffs = coeffs[:]
    
    # Also try some structured candidates
    structured_candidates = [
        # Consecutive
        [1]*12,
        # Alternating signs
        [1 if i%2==0 else -1 for i in range(12)],
        # Powers of 2 exponents
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]
    
    # Try: exponents at 0,1,2,4,8,16,32 (geometric) - but that's only 7, need 12
    # Try: 0,1,2,3,4,5,6,7,8,9,10,11 (consecutive) - already done
    # Try: 0,1,2,4,5,6,8,9,10,12,13,14
    geo_exponents = [0,1,2,4,5,6,8,9,10,12,13,14]
    if len(geo_exponents) == 12:
        coeffs = [0] * (geo_exponents[-1] + 1)
        for e in geo_exponents:
            coeffs[e] = 1
        if _valid_candidate(coeffs):
            sq = _square_poly(coeffs)
            sq_nonzero = _count_nonzero(sq)
            if sq_nonzero < best_square_nonzero:
                best_square_nonzero = sq_nonzero
                best_coeffs = coeffs[:]
    
    # Try: 0,1,2,3,4,5,6,7,8,9,10,20
    mixed_exponents = [0,1,2,3,4,5,6,7,8,9,10,20]
    coeffs = [0] * (mixed_exponents[-1] + 1)
    for e in mixed_exponents:
        coeffs[e] = 1
    if _valid_candidate(coeffs):
        sq = _square_poly(coeffs)
        sq_nonzero = _count_nonzero(sq)
        if sq_nonzero < best_square_nonzero:
            best_square_nonzero = sq_nonzero
            best_coeffs = coeffs[:]
    
    # Try with negative coefficients to get cancellation
    # P(x) = 1 + x - x^2 - x^3 + x^4 + x^5 - ... 
    for sign_pattern in range(64):
        if time.time() > deadline:
            break
        coeffs = [0] * 12
        for i in range(12):
            coeffs[i] = 1 if (sign_pattern >> i) & 1 else -1
        if _valid_candidate(coeffs):
            sq = _square_poly(coeffs)
            sq_nonzero = _count_nonzero(sq)
            if sq_nonzero < best_square_nonzero:
                best_square_nonzero = sq_nonzero
                best_coeffs = coeffs[:]
    
    # Trim best_coeffs
    while len(best_coeffs) > 0 and best_coeffs[-1] == 0:
        best_coeffs.pop()
    
    return best_coeffs
# EVOLVE-BLOCK-END
