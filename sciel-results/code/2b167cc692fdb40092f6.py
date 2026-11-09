"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    n = len(c)
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid_poly(c):
    """Check if polynomial is valid: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(c) == 0 or len(c) > 41:
        return False
    if c[-1] == 0:
        return False
    if any(abs(x) > 64 for x in c):
        return False
    if sum(1 for x in c if x != 0) != 12:
        return False
    return True

def _local_search(poly, deadline, best_sq_nnz, best_poly):
    """
    Local search: perturb coefficients by small amounts, check validity, evaluate square nnz.
    Returns updated (best_sq_nnz, best_poly).
    """
    c = poly[:]
    current_sq_nnz = best_sq_nnz
    
    # Precompute square for current
    sq = _square_coeffs(c)
    cur_nnz = _nnz(sq)
    if cur_nnz < current_sq_nnz:
        current_sq_nnz = cur_nnz
        best_poly = c[:]
    
    # Local search: try flipping signs, incrementing/decrementing by 1,2,3
    steps = [1, -1, 2, -2, 3, -3]
    improved = True
    max_rounds = 200
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try each position
        for i in range(len(c)):
            if time.time() > deadline:
                break
            
            old_val = c[i]
            for step in steps:
                new_val = old_val + step
                if new_val == 0:
                    # Allow zero only if we still have 12 nonzero terms after change
                    # Actually, if old_val != 0 and new_val == 0, we lose a term
                    # If old_val == 0 and new_val != 0, we gain a term
                    pass
                if abs(new_val) > 64:
                    continue
                if new_val == old_val:
                    continue
                
                c[i] = new_val
                # Check validity quickly
                if not _is_valid_poly(c):
                    c[i] = old_val
                    continue
                
                sq = _square_coeffs(c)
                nnz_sq = _nnz(sq)
                
                if nnz_sq < current_sq_nnz:
                    current_sq_nnz = nnz_sq
                    best_poly = c[:]
                    improved = True
                else:
                    c[i] = old_val
    
    return current_sq_nnz, best_poly

def _structured_candidates():
    """Generate structured candidate polynomials likely to have sparse squares."""
    candidates = []
    
    # 1. All ones degree 11 (palindromic)
    candidates.append([1]*12)
    
    # 2. Alternating signs
    candidates.append([1,-1,1,-1,1,-1,-1,1,-1,1,-1,1])
    
    # 3. Palindromic patterns with small coefficients
    # Degree 11 palindromic: c0..c5 free, c6..c11 = c5..c0
    patterns = [
        [1,1,1,1,1,1],
        [1,1,1,-1,-1,-1],
        [1,-1,1,-1,1,-1],
        [1,2,1,2,1,2],
        [2,1,2,1,2,1],
        [1,1,-1,1,1,-1],
        [1,-1,-1,-1,-1,1],
        [3,1,1,1,1,3],
        [1,3,1,1,3,1],
        [1,1,3,1,1,3],
        [1,2,3,2,1,2],
        [2,1,2,3,1,2],
        [1,1,1,2,1,1],
        [1,2,1,1,2,1],
        [1,1,2,1,1,2],
    ]
    for p in patterns:
        pal = p + [p[4], p[3], p[2], p[1], p[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    # 4. Try polynomials of the form sum of specific monomials
    # x^11 + x^10 + ... with gaps
    # For 12 terms, try sparse support sets
    import itertools
    
    # Support sets of size 12 within degree 0..40
    # Try palindromic supports
    # Degree 11: all 12 positions used
    # Degree 13: 14 positions, need 12 nonzero -> 2 zeros
    # Try specific gap patterns
    
    # P(x) = x^13 + x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
    # (missing x^3 and x^1)
    supports = [
        [13,12,11,10,9,8,7,6,5,4,2,0],
        [13,12,11,10,9,8,7,6,5,3,1,0],
        [13,12,11,10,9,8,7,6,4,2,1,0],
        [13,12,11,10,9,8,7,5,4,3,2,1],
        [13,12,11,10,9,8,6,5,4,3,2,1],
        [13,12,11,10,9,7,6,5,4,3,2,1],
        [13,12,11,10,8,7,6,5,4,3,2,1],
        [13,12,11,9,8,7,6,5,4,3,2,1],
        [13,12,10,9,8,7,6,5,4,3,2,1],
        [13,11,10,9,8,7,6,5,4,3,2,1],
    ]
    for sup in supports:
        poly = [0]*14
        for d in sup:
            poly[d] = 1
        if _is_valid_poly(poly):
            candidates.append(poly)
    
    # 5. Try with coefficient -1 for some terms
    for sup in supports[:5]:
        poly = [0]*14
        for j, d in enumerate(sup):
            poly[d] = 1 if j % 2 == 0 else -1
        if _is_valid_poly(poly):
            candidates.append(poly)
    
    # 6. Try (1+x+x^2+x^3+x^4+x^5)^2 type structures
    # Actually, let's try P(x) = Q(x)^2 where Q has 6 terms... no, that gives P with up to 36 terms
    # Better: P(x) = Q(x)*R(x) where Q and R have few terms
    
    # 7. Try degree 15 palindromic with 12 nonzero terms
    # c0..c15, palindromic c_i = c_{15-i}
    # 16 positions, 8 independent, need 12 nonzero total -> 4 zeros among the 16
    # Try specific patterns
    pal15_patterns = [
        [1,0,1,0,1,0,1,0],  # c0,c2,c4,c6 nonzero, c1,c3,c5,c7 zero
        [0,1,0,1,0,1,0,1],  # c1,c3,c5,c7 nonzero
        [1,1,0,0,1,1,0,0],
        [1,0,0,1,0,0,1,0],
        [1,1,1,0,1,1,1,0],
    ]
    for p in pal15_patterns:
        pal = p + [p[6], p[5], p[4], p[3], p[2], p[1], p[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    # 8. Try Chebyshev-like or other structured polynomials
    # T_6(x) = 32x^6 - 48x^4 + 18x^2 - 1 (4 terms)
    # Try products or sums
    
    # 9. Try P(x) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    # with some coefficients as -1 to create cancellation
    
    # 10. Random palindromic with coefficients from {-2,-1,1,2}
    rng = random.Random(12345)
    for _ in range(200):
        free = [rng.choice([-2,-1,1,2]) for _ in range(6)]
        pal = free + [free[4], free[3], free[2], free[1], free[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    
    # Phase 1: Evaluate structured candidates
    candidates = _structured_candidates()
    for c in candidates:
        if time.time() > deadline:
            break
        if not _is_valid_poly(c):
            continue
        sq = _square_coeffs(c)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = c[:]
    
    # Phase 2: Random search on palindromic degree-11 polynomials
    # 6 independent coefficients in range [-8, 8], all must be nonzero
    non_zero_vals = list(range(-8, 9))
    non_zero_vals.remove(0)
    
    rng = random.Random(42)
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 50000 == 0 and time.time() > deadline:
            break
        
        # Pick 6 nonzero coefficients
        free = [rng.choice(non_zero_vals) for _ in range(6)]
        # Build palindromic degree 11: c0..c5 = free, c6..c11 = free[5],free[4],...,free[0]
        poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        
        if not _is_valid_poly(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # Phase 3: Local search from best found
    if time.time() < deadline:
        best_sq_nnz, best_poly = _local_search(best_poly, deadline, best_sq_nnz, best_poly)
    
    # Phase 4: Additional random search with different structure
    # Try non-palindromic polynomials with 12 terms
    if time.time() < deadline:
        # Choose 12 distinct degrees from 0..40
        for _ in range(10000):
            if time.time() > deadline:
                break
            degrees = rng.sample(range(0, 41), 12)
            degrees.sort()
            if degrees[-1] == 0:
                continue
            coeffs = [rng.choice([-2,-1,1,2,3]) for _ in range(12)]
            poly = [0]*41
            for d, c in zip(degrees, coeffs):
                poly[d] = c
            # Trim trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            if not _is_valid_poly(poly):
                continue
            sq = _square_coeffs(poly)
            nnz_sq = _nnz(sq)
            if nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = poly[:]
    
    # Final local search
    if time.time() < deadline:
        best_sq_nnz, best_poly = _local_search(best_poly, deadline, best_sq_nnz, best_poly)
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly

# EVOLVE-BLOCK-END
