import time
import random
from itertools import combinations, product

def _square(coeffs):
    """Compute exact integer square of polynomial given as coefficient list (ascending degree)."""
    n = len(coeffs)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    if not nz:
        return [0]
    max_deg = 2 * (n - 1)
    result = [0] * (max_deg + 1)
    for i, ci in nz:
        for j, cj in nz:
            result[i + j] += ci * cj
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _is_valid(coeffs):
    if len(coeffs) > 41 or len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if _nonzero_count(coeffs) != 12:
        return False
    return True

def _score(coeffs):
    sq = _square(coeffs)
    return _nonzero_count(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s kill)
    
    best_poly = None
    best_score = float('inf')
    
    def update_best(poly):
        nonlocal best_poly, best_score
        if _is_valid(poly):
            s = _score(poly)
            if s < best_score:
                best_score = s
                best_poly = poly[:]
    
    # Key insight: If P = sum c_i * x^{e_i}, then P^2 has terms at positions e_i + e_j.
    # To minimize nonzero terms in P^2, we want many of these sums to coincide (collision)
    # and we want the coefficients to cancel out.
    # 
    # Strategy: Use a structure where the exponents form an arithmetic progression or
    # have high collision rates. The most effective structure for cancellation is
    # P = A + B where A and B are "orthogonal" in some sense, or using symmetric
    # structures like P = x^k * (x^m - 1) * Q where Q is chosen to have few terms in its square.
    #
    # A very strong candidate: P with exponents forming a "Golay-like" or "bent" sequence
    # pattern, but simpler: use the fact that (x^a + x^b + ...)^2 can have cancellations
    # if we use coefficients like 1, -1 in a balanced way.
    #
    # Let's try: P = sum_{i=0}^{11} c_i * x^{d_i} where d_i are chosen so that the
    # multiset {d_i + d_j} has few distinct values, and the coefficients are chosen
    # so that for each distinct sum value, the total coefficient is 0 or very few are nonzero.
    #
    # The theoretical minimum: If all 12*12=144 terms of P^2 collapse to 1 term, we'd have
    # a perfect square of a monomial, which is impossible with 12 terms.
    # Next best: P^2 has k nonzero terms. We want to minimize k.
    #
    # A known construction: If P is a "bent" polynomial over GF(2), its square is a monomial.
    # But we're in integers. However, we can use the structure:
    # P = x^a * (1 + x + x^2 + ... + x^{2^m-1}) type structures don't work directly.
    #
    # Let's use a more systematic approach:
    # 1. Try polynomials where exponents are in an arithmetic progression: d, d+1, ..., d+11
    #    and coefficients are chosen from {1, -1} to maximize cancellation.
    # 2. Try exponents with step 2: d, d+2, ..., d+22
    # 3. Try the "product of two 6-term polynomials" approach: P = Q * R where Q and R
    #    each have 6 terms, and Q*R has 12 terms (no cross-term collisions within the product
    #    that create extra terms), and P^2 = Q^2 * R^2 has few terms.
    #
    # Actually, the most promising approach for a JUMP: 
    # Use the identity that if P has a "self-convolution" structure, P^2 can be sparse.
    # Specifically, try P where the exponent set is a "sum-free" set or has high additive structure.
    #
    # Let's focus on: P with exponents forming a geometric-like or highly structured set,
    # and use local search to find optimal coefficients.
    
    random.seed(12345)
    
    # Phase 1: Structured candidates
    # Try arithmetic progressions with various steps and sign patterns
    for step in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        max_base = 40 - step * 11
        if max_base < 0:
            continue
        for base in range(0, max_base + 1):
            # Try all sign patterns with ±1 coefficients
            # Use a smarter search: try patterns with high cancellation potential
            for signs in [
                [1]*12,
                [1,-1]*6,
                [1,1,-1,-1]*3,
                [1,-1,1,-1]*6,
                [1,-1,-1,1]*3,
                [1,1,1,-1,-1,-1]*2,
                [1,-1,1,1,-1,-1]*2,
            ]:
                poly = [0] * 41
                for i in range(12):
                    poly[base + i * step] = signs[i]
                if poly[-1] != 0:
                    update_best(poly)
                if time.time() > deadline:
                    return best_poly if best_poly else [1]*12
    
    # Phase 2: Try the "product" structure more carefully
    # P = sum_{i=0}^{5} a_i x^{p_i} * sum_{j=0}^{5} b_j x^{q_j}
    # where {p_i} and {q_j} are chosen so that the 36 products p_i + q_j 
    # give exactly 12 distinct values (each appearing 3 times), and the
    # coefficients a_i * b_j are chosen to cancel within each group.
    
    # A simpler high-value strategy: Use local search starting from good structured seeds
    # with a much more aggressive neighborhood.
    
    def local_search(poly, max_time=5.0):
        """Aggressive local search with multiple move types."""
        poly = poly[:]
        if not _is_valid(poly):
            return poly, _score(poly) if _is_valid(poly) else float('inf')
        
        current_score = _score(poly)
        start = time.time()
        
        while time.time() - start < max_time:
            improved = False
            nz = [(i, c) for i, c in enumerate(poly) if c != 0]
            
            # Move type 1: Change a coefficient
            for idx, old_val in nz:
                for new_val in [-64, -32, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 32, 64]:
                    if new_val == old_val:
                        continue
                    poly[idx] = new_val
                    s = _score(poly)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        poly[idx] = old_val
            
            # Move type 2: Flip sign
            for idx, old_val in nz:
                poly[idx] = -old_val
                s = _score(poly)
                if s < current_score:
                    current_score = s
                    improved = True
                else:
                    poly[idx] = old_val
            
            # Move type 3: Move a term to a nearby position
            nz = [(i, c) for i, c in enumerate(poly) if c != 0]
            for idx, val in nz:
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_idx = idx + delta
                    if new_idx < 0 or new_idx >= 41:
                        continue
                    if poly[new_idx] != 0:
                        continue
                    poly[idx] = 0
                    poly[new_idx] = val
                    # Trim
                    while len(poly) > 1 and poly[-1] == 0:
                        poly.pop()
                    if _is_valid(poly):
                        s = _score(poly)
                        if s < current_score:
                            current_score = s
                            improved = True
                        else:
                            poly[new_idx] = 0
                            poly[idx] = val
                    else:
                        poly[new_idx] = 0
                        poly[idx] = val
            
            # Move type 4: Swap two coefficients
            nz = [(i, c) for i, c in enumerate(poly) if c != 0]
            for i in range(len(nz)):
                for j in range(i+1, len(nz)):
                    idx1, val1 = nz[i]
                    idx2, val2 = nz[j]
                    poly[idx1] = val2
                    poly[idx2] = val1
                    s = _score(poly)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        poly[idx1] = val1
                        poly[idx2] = val2
            
            if not improved:
                break
        
        return poly, current_score
    
    # Phase 3: Random search with local improvement
    while time.time() < deadline:
        # Generate random polynomial
        degrees = sorted(random.sample(range(41), 12))
        poly = [0] * 41
        for d in degrees:
            poly[d] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -8, 8, -16, 16])
        if poly[40] == 0:
            poly[degrees[-1]] = random.choice([-1, 1, 2, 3, 4])
        last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
        poly = poly[:last_nz+1]
        
        if not _is_valid(poly):
            continue
        
        s = _score(poly)
        if s < best_score:
            best_score = s
            best_poly = poly[:]
        
        # Local search on promising candidates
        if s <= best_score + 2:
            improved_poly, improved_score = local_search(poly, max_time=3.0)
            if _is_valid(improved_poly) and improved_score < best_score:
                best_score = improved_score
                best_poly = improved_poly[:]
        
        if time.time() > deadline - 5:
            break
    
    if best_poly is not None:
        return best_poly
    
    # Fallback
    fallback = [0] * 41
    for i in range(12):
        fallback[i] = 1
    return fallback
