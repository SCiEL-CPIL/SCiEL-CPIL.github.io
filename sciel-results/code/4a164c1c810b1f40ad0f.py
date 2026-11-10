import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    best_score = float('inf')
    best_poly = None

    # Strategy 1: Structured decomposition with pre-optimized blocks
    # A 12-term polynomial squared can have at most 12*13/2 = 78 nonzero terms if no cancellation.
    # We want to minimize the number of nonzero terms in the square.
    # Key insight: if we can pair terms so that cross-terms cancel, we reduce the count.
    
    # Let's try a structured approach:
    # P(x) = sum_{i=0}^{11} a_i * x^{e_i} where e_i are distinct exponents
    # P(x)^2 = sum a_i^2 x^{2e_i} + sum_{i<j} 2*a_i*a_j x^{e_i+e_j}
    # To minimize nonzero terms, we want many e_i+e_j to coincide (collision) and cancel.
    
    # A powerful structure: use exponents in arithmetic progression with specific signs
    # For example, if exponents are {0, 1, 2, ..., 11}, then e_i+e_j ranges from 0 to 22.
    # The number of distinct sums is at most 23, but with 78 cross-terms, many collisions occur.
    
    # Better: use symmetric exponents to force more cancellation.
    # Let's try: exponents = {0, 1, 3, 7, 15, 31, 31, 15, 7, 3, 1, 0} - no, must be distinct.
    
    # Actually, let's try a different approach: use a known sparse square construction.
    # If P(x) = x^6 + x^5 - x^4 - x^3 - x^2 - x + 1, let's check its square.
    
    # Let me compute some candidates systematically.
    
    def count_nonzero(poly):
        """Count nonzero coefficients in a polynomial represented as list."""
        return sum(1 for c in poly if c != 0)
    
    def poly_square(coeffs):
        """Compute the exact square of a polynomial given as coefficient list (ascending degree)."""
        n = len(coeffs)
        result = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    result[2*i] += val
                else:
                    result[i+j] += 2 * val
        # Trim trailing zeros but keep at least 1 element
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    def validate(poly):
        """Check if poly meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) == 0 or poly[-1] == 0:
            return False
        for c in poly:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        return True
    
    # Generate candidate polynomials with 12 nonzero terms
    # Strategy: use random supports and try to find good coefficient assignments
    
    # First, try some structured candidates
    structured_candidates = []
    
    # Candidate type 1: symmetric exponents
    # P(x) = sum a_i x^{e_i} where exponents are chosen to maximize collisions in e_i+e_j
    # Optimal: use exponents that form a "sum-free" structure minimally
    
    # Let's try exponents: 0,1,2,3,4,5,6,7,8,9,10,11 (arithmetic progression)
    # With signs: +1, -1, +1, -1, ... to encourage cancellation
    
    # Actually, let's be more systematic. Try all sign patterns for small exponent sets.
    
    def try_candidate(exponents, coefficients):
        """Try a specific polynomial and return its score."""
        max_deg = max(exponents)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            poly[e] += c
        
        if not validate(poly):
            return None
        
        sq = poly_square(poly)
        # Check that square coefficients are reasonable
        for c in sq:
            if abs(c) > 10000:  # sanity check
                return None
        
        score = count_nonzero(sq)
        return score, poly, sq
    
    # Try structured candidates
    # Type A: consecutive exponents with alternating signs
    for start in range(0, 5):
        exponents = [start + i for i in range(12)]
        for signs in [
            [1]*12,
            [1,-1]*6,
            [1,1,-1,-1]*3,
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
            [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
            [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        ]:
            coeffs = [s for s in signs]
            result = try_candidate(exponents, coeffs)
            if result:
                score, poly, sq = result
                if score < best_score:
                    best_score = score
                    best_poly = poly
                    if best_score <= 12:
                        break
        if best_score <= 12:
            break
    
    if best_score <= 12:
        if time.time() > deadline:
            return best_poly
        # Can we do better than 12? Try more aggressively.
    
    # Type B: Use exponents with more structure
    # Try: 0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16 (gaps of 1 and 2)
    for exp_set in [
        [0,1,2,4,5,7,8,10,11,13,14,16],
        [0,1,3,4,6,7,9,10,12,13,15,16],
        [0,2,4,6,8,10,12,14,16,18,20,22],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,10,20],
    ]:
        if len(exp_set) != 12:
            continue
        for signs in [
            [1]*12,
            [1,-1]*6,
            [1,1,-1,-1]*3,
            [1,-1,1,-1]*3,
            [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
        ]:
            coeffs = signs[:]
            result = try_candidate(exp_set, coeffs)
            if result:
                score, poly, sq = result
                if score < best_score:
                    best_score = score
                    best_poly = poly
    
    if time.time() > deadline:
        return best_poly
    
    # Random search with local improvement
    random.seed(42)
    
    def random_poly():
        """Generate a random valid polynomial."""
        while True:
            # Choose 12 distinct exponents from 0..40
            exponents = sorted(random.sample(range(41), 12))
            # Choose coefficients from [-64, 64] excluding 0
            coefficients = []
            for _ in range(12):
                c = random.choice([i for i in range(-64, 65) if i != 0])
                coefficients.append(c)
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coefficients):
                poly[e] = c
            if validate(poly):
                return poly
    
    # Local search: try to improve by changing one coefficient or exponent
    def local_search(poly, max_steps=100):
        current = poly[:]
        sq = poly_square(current)
        current_score = count_nonzero(sq)
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try changing one coefficient
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_val = current[i]
                for new_val in [old_val + 1, old_val - 1, -old_val, old_val + 2, old_val - 2]:
                    if new_val == 0 or new_val < -64 or new_val > 64:
                        continue
                    current[i] = new_val
                    if not validate(current):
                        current[i] = old_val
                        continue
                    sq_new = poly_square(current)
                    score_new = count_nonzero(sq_new)
                    if score_new < current_score:
                        current_score = score_new
                        improved = True
                        break
                    elif score_new == current_score and random.random() < 0.1:
                        # Accept with small probability for diversity
                        current_score = score_new
                        improved = True
                        break
                    else:
                        current[i] = old_val
            
            if not improved:
                # Try moving one term to a different exponent
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_exp = i
                    old_val = current[i]
                    # Try moving to a nearby empty position
                    for new_exp in [old_exp-1, old_exp+1, old_exp-2, old_exp+2]:
                        if new_exp < 0 or new_exp >= 41:
                            continue
                        if current[new_exp] != 0:
                            continue
                        current[old_exp] = 0
                        current[new_exp] = old_val
                        if not validate(current):
                            current[old_exp] = old_val
                            current[new_exp] = 0
                            continue
                        sq_new = poly_square(current)
                        score_new = count_nonzero(sq_new)
                        if score_new < current_score:
                            current_score = score_new
                            improved = True
                            break
                        else:
                            current[old_exp] = old_val
                            current[new_exp] = 0
            
            if not improved:
                break
        
        return current, current_score
    
    # Run multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        poly = random_poly()
        sq = poly_square(poly)
        score = count_nonzero(sq)
        
        if score < best_score:
            best_score = score
            best_poly = poly
        
        # Local search
        poly_ls, score_ls = local_search(poly, max_steps=50)
        if score_ls < best_score:
            best_score = score_ls
            best_poly = poly_ls
        
        if best_score <= 14:
            break
    
    if time.time() > deadline:
        return best_poly
    
    # Try a more targeted approach: search for polynomials where the square has very few terms
    # Key insight: if P(x) = A(x) + B(x) where A and B have disjoint supports and 
    # AB has all coefficients cancel, then P^2 = A^2 + B^2.
    # This is hard, but let's try: P(x) = sum of 6 terms with exponents in [0,5] and 6 terms in [10,15]
    # such that cross-terms cancel.
    
    # Actually, a simpler idea: use P(x) where the exponents form pairs (e, f) such that 
    # e+f is the same for multiple pairs, forcing collisions.
    
    # Let's try: exponents = {0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6}
    # Then e_i + e_j for i<j includes many pairs summing to 11.
    # Specifically: 0+11=11, 1+10=11, 2+9=11, 3+8=11, 4+7=11, 5+6=11
    # So 6 cross-terms all go to x^11! If we choose signs so they cancel, great.
    
    exps = [0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
    # Pairs summing to 11: (0,11), (1,10), (2,9), (3,8), (4,7), (5,6)
    # Coefficients a0*a11, a1*a10, a2*a9, a3*a8, a4*a7, a5*a6
    # For cancellation at x^11: a0*a11 + a1*a10 + a2*a9 + a3*a8 + a4*a7 + a5*a6 = 0
    
    # Other sums: 0+1=1, 0+2=2, ..., 0+6=6, 11+1=12, 11+2=13, ..., 11+6=17
    # 1+2=3, 1+3=4, ..., 1+6=7, 10+2=12, 10+3=13, ..., 10+6=16
    # 2+3=5, ..., 2+6=8, 9+3=12, ..., 9+6=15
    # 3+4=7, ..., 3+6=9, 8+4=12, ..., 8+6=14
    # 4+5=9, 4+6=10, 7+5=12, 7+6=13
    # 5+6=11 (already counted)
    
    # Let's try: a0=1, a11=1, a1=1, a10=-1, a2=1, a9=1, a3=-1, a8=1, a4=1, a7=-1, a5=1, a6=-1
    # Check x^11: 1*1 + 1*(-1) + 1*1 + (-1)*1 + 1*(-1) + 1*(-1) = 1-1+1-1-1-1 = -2. Not zero.
    
    # Try: a0=1, a11=1, a1=1, a10=-1, a2=1, a9=-1, a3=1, a8=-1, a4=1, a7=-1, a5=1, a6=-1
    # x^11: 1*1 + 1*(-1) + 1*(-1) + 1*(-1) + 1*(-1) + 1*(-1) = 1-1-1-1-1-1 = -4. No.
    
    # Try: a0=1, a11=-1, a1=1, a10=1, a2=1, a9=-1, a3=1, a8=1, a4=1, a7=-1, a5=1, a6=1
    # x^11: 1*(-1) + 1*1 + 1*(-1) + 1*1 + 1*(-1) + 1*1 = -1+1-1+1-1+1 = 0! 
    
    coeffs_try = [1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1]
    # Map to exponents: exps[0]=0->1, exps[1]=11->-1, exps[2]=1->1, exps[3]=10->1, 
    # exps[4]=2->1, exps[5]=9->-1, exps[6]=3->1, exps[7]=8->1, exps[8]=4->1, exps[9]=7->-1,
    # exps[10]=5->1, exps[11]=6->1
    poly_try = [0] * 12
    for e, c in zip(exps, coeffs_try):
        poly_try[e] = c
    
    if validate(poly_try):
        sq_try = poly_square(poly_try)
        score_try = count_nonzero(sq_try)
        if score_try < best_score:
            best_score = score_try
            best_poly = poly_try
    
    # Try variations of this pattern
    for variant in range(20):
        if time.time() > deadline:
            break
        exps_v = [0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
        coeffs_v = []
        # Random signs with constraint that sum of products for pairs summing to 11 is 0
        # Pairs: (0,11), (1,10), (2,9), (3,8), (4,7), (5,6)
        # Let's pick 5 signs randomly and determine the 6th
        for _ in range(5):
            coeffs_v.append(random.choice([1, -1]))
        # coeffs_v[0]=a0, coeffs_v[1]=a11, coeffs_v[2]=a1, coeffs_v[3]=a10, coeffs_v[4]=a2, coeffs_v[5]=a9
        # Need: a0*a11 + a1*a10 + a2*a9 + a3*a8 + a4*a7 + a5*a6 = 0
        # a3, a4, a5, a6, a7, a8 are not yet set. Let's set them.
        # Actually this is getting complex. Let me just try random sign patterns.
        coeffs_v = [random.choice([1, -1]) for _ in range(12)]
        poly_v = [0] * 12
        for e, c in zip(exps_v, coeffs_v):
            poly_v[e] = c
        if validate(poly_v):
            sq_v = poly_square(poly_v)
            score_v = count_nonzero(sq_v)
            if score_v < best_score:
                best_score = score_v
                best_poly = poly_v
    
    if time.time() > deadline:
        return best_poly
    
    # Final fallback: return the best we found
    if best_poly is None:
        # Default fallback
        best_poly = [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
