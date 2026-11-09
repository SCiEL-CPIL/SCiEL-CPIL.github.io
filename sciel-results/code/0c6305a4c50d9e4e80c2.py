import time
import itertools
import random

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _square_coeffs(poly):
    """Compute exact integer square of polynomial given as list of (degree, coeff)."""
    n = len(poly)
    max_deg = 0
    for d, c in poly:
        if d > max_deg:
            max_deg = d
    sq_len = 2 * max_deg + 1
    sq = [0] * sq_len
    for i in range(n):
        di, ci = poly[i]
        for j in range(i, n):
            dj, cj = poly[j]
            prod = ci * cj
            deg = di + dj
            if i == j:
                sq[deg] += prod
            else:
                sq[deg] += 2 * prod
    # Trim trailing zeros
    while len(sq) > 1 and sq[-1] == 0:
        sq.pop()
    return sq

def _build_poly(positions, signs):
    """Build polynomial from positions and signs."""
    poly = []
    for pos, sign in zip(positions, signs):
        if sign != 0:
            poly.append((pos, sign))
    return poly

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    best_sq_count = None
    best_result = None  # (sq_count, poly_list)
    
    # First, try some known good structures
    # Strategy: use palindromic polynomials with ±1 coefficients
    
    # Try: 12 terms, all ±1, search over sign patterns for fixed support
    # Start with a well-chosen support pattern
    
    # Support: try consecutive or symmetric patterns
    # For degree <= 40, positions must be in [0, 40]
    
    # Try a symmetric support: positions symmetric around center
    # 12 terms: 6 pairs + maybe 2 center terms
    
    # Let's try several support patterns and search signs
    
    # Pattern 1: positions 0,1,2,3,4,5, 35,36,37,38,39,40 (two blocks)
    # Pattern 2: positions 0,1,2,3,4,5,6, 34,35,36,37,38,39
    # Pattern 3: spread out
    
    support_patterns = [
        # Two symmetric blocks
        [0,1,2,3,4,5, 35,36,37,38,39,40],
        [0,1,2,3,4,5,6, 34,35,36,37,38,39],
        [0,1,2,3,4,5, 34,35,36,37,38,39],
        # More spread
        [0,2,4,6,8,10, 30,32,34,36,38,40],
        [0,3,6,9,12,15, 25,28,31,34,37,40],
        # Palindromic: a_i = a_{d-i}
        [0,1,2,3,4,5, 35,36,37,38,39,40],  # palindromic if signs match
        [0,2,4,6,8,10, 30,32,34,36,38,40],
    ]
    
    # Also try: single block with gaps
    support_patterns.append(list(range(12)))  # 0..11
    support_patterns.append(list(range(29, 41)))  # 29..40
    
    # For each support pattern, search sign assignments (±1 for each term)
    # 2^12 = 4096 sign patterns per support - very fast
    
    for support in support_patterns:
        if time.time() > deadline:
            break
        if len(support) != 12:
            continue
        if max(support) > 40:
            continue
        if len(set(support)) != 12:
            continue
        
        # Try all 2^12 sign patterns
        for mask in range(4096):
            if time.time() > deadline:
                break
            signs = []
            for i in range(12):
                if (mask >> i) & 1:
                    signs.append(1)
                else:
                    signs.append(-1)
            
            poly = _build_poly(support, signs)
            sq = _square_coeffs(poly)
            sq_count = _count_nonzero(sq)
            
            if best_sq_count is None or sq_count < best_sq_count:
                best_sq_count = sq_count
                # Convert to coefficient list
                max_deg = max(d for d, c in poly)
                coeff_list = [0] * (max_deg + 1)
                for d, c in poly:
                    coeff_list[d] = c
                best_result = (sq_count, coeff_list, poly, support, signs)
    
    # Now try random searches with varied supports
    if best_result is None:
        # Fallback
        best_result = (None, [1]*12, None, None, None)
    
    # Random search with diverse supports
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate random support: 12 distinct positions in [0, 40]
        support = sorted(random.sample(range(41), 12))
        signs = [random.choice([1, -1]) for _ in range(12)]
        
        poly = _build_poly(support, signs)
        sq = _square_coeffs(poly)
        sq_count = _count_nonzero(sq)
        
        if best_result is None or best_sq_count is None or sq_count < best_sq_count:
            best_sq_count = sq_count
            max_deg = max(d for d, c in poly)
            coeff_list = [0] * (max_deg + 1)
            for d, c in poly:
                coeff_list[d] = c
            best_result = (sq_count, coeff_list, poly, support, signs)
        
        # Also try palindromic variants: mirror the support
        d_max = 40
        pal_support = sorted(set([min(p, d_max - p) for p in support]))
        if len(pal_support) == 12:
            pal_signs = [random.choice([1, -1]) for _ in range(12)]
            pal_poly = _build_poly(pal_support, pal_signs)
            pal_sq = _square_coeffs(pal_poly)
            pal_sq_count = _count_nonzero(pal_sq)
            if best_sq_count is None or pal_sq_count < best_sq_count:
                best_sq_count = pal_sq_count
                max_deg = max(d for d, c in pal_poly)
                coeff_list = [0] * (max_deg + 1)
                for d, c in pal_poly:
                    coeff_list[d] = c
                best_result = (pal_sq_count, coeff_list, pal_poly, pal_support, pal_signs)
    
    # Return the best coefficient list found
    if best_result is not None and len(best_result) >= 2:
        return best_result[1]
    else:
        # Fallback: [1]*12
        return [1] * 12
