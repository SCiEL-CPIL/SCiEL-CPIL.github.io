import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_poly = None
    best_score = 100  # max possible nonzero terms in square is 23
    
    def count_nonzero_square(coeffs):
        """Count nonzero terms in the square of polynomial given by coeffs list."""
        n = len(coeffs)
        # Square coefficients: degree 0 to 2*(n-1)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                deg = i + j
                val = coeffs[i] * coeffs[j]
                if i != j:
                    val *= 2
                sq[deg] += val
        # Trim trailing zeros but keep track
        count = 0
        for v in sq:
            if v != 0:
                count += 1
        return count
    
    def evaluate(poly):
        """Poly is list of (exp, coeff) pairs. Returns score (nonzero terms in square)."""
        if len(poly) != 12:
            return 100
        max_exp = max(e for e, c in poly)
        if max_exp >= 41:
            return 100
        # Build coefficient array
        coeffs = [0] * (max_exp + 1)
        for e, c in poly:
            if c == 0:
                return 100
            coeffs[e] = c
        # Trim to remove trailing zeros
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        if coeffs[-1] == 0:
            return 100
        if len(coeffs) > 41:
            return 100
        # Check all entries in [-64, 64]
        for c in coeffs:
            if c < -64 or c > 64:
                return 100
        return count_nonzero_square(coeffs)
    
    def poly_to_list(poly):
        """Convert list of (exp, coeff) to coefficient list."""
        max_exp = max(e for e, c in poly)
        coeffs = [0] * (max_exp + 1)
        for e, c in poly:
            coeffs[e] = c
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    # Strategy 1: Try structured patterns
    # Arithmetic progressions
    for start in range(0, 5):
        for step in range(1, 8):
            if time.time() > deadline:
                break
            exponents = [start + i * step for i in range(12)]
            if max(exponents) >= 41:
                continue
            # Try various coefficient patterns
            for coeff_pattern in [
                [1]*12, [-1]*12,
                [1,-1]*6, [1,1,-1,-1]*3,
                [1,0,1,0,1,0,1,0,1,0,1,0],  # skip, need 12 nonzero
            ]:
                if len([c for c in coeff_pattern if c != 0]) != 12:
                    continue
                poly = list(zip(exponents, coeff_pattern))
                score = evaluate(poly)
                if score < best_score:
                    best_score = score
                    best_poly = poly[:]
            
            # Try random coefficients
            for _ in range(200):
                if time.time() > deadline:
                    break
                coeffs = [random.choice([-2,-1,1,2,3,4,5]) for _ in range(12)]
                poly = list(zip(exponents, coeffs))
                score = evaluate(poly)
                if score < best_score:
                    best_score = score
                    best_poly = poly[:]
    
    # Strategy 2: Random search with local improvement
    def random_poly():
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coeffs = [random.choice([-4,-3,-2,-1,1,2,3,4]) for _ in range(12)]
        return list(zip(exponents, coeffs))
    
    def local_search(poly, max_iters=200):
        current = poly[:]
        current_score = evaluate(current)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Random move
            new_poly = current[:]
            move_type = random.random()
            if move_type < 0.4:
                # Change a coefficient
                idx = random.randint(0, 11)
                new_coeff = random.choice([-6,-4,-3,-2,-1,1,2,3,4,6])
                if new_coeff == 0:
                    new_coeff = 1
                new_poly[idx] = (new_poly[idx][0], new_coeff)
            elif move_type < 0.7:
                # Change an exponent
                idx = random.randint(0, 11)
                new_exp = random.randint(0, 40)
                # Check for duplicate exponents
                other_exps = [new_poly[i][0] for i in range(12) if i != idx]
                if new_exp not in other_exps:
                    new_poly[idx] = (new_exp, new_poly[idx][1])
                    new_poly.sort(key=lambda x: x[0])
                else:
                    continue
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_poly[i], new_poly[j] = new_poly[j], new_poly[i]
                new_poly.sort(key=lambda x: x[0])
            
            new_score = evaluate(new_poly)
            if new_score < current_score:
                current = new_poly
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current[:]
        return current, current_score
    
    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 5.0:
        poly = random_poly()
        poly, score = local_search(poly, max_iters=100)
        restarts += 1
        if score < best_score:
            best_score = score
            best_poly = poly[:]
    
    # Strategy 3: Targeted search for known good structures
    # Try polynomials with symmetric exponent patterns
    # e.g., exponents: 0, a, b, c, d, e, f, g, h, i, j, k where some symmetry exists
    
    # Try: p(x) = sum of x^{e_i} with e_i forming a Sidon-like set but with deliberate collisions
    # Key: if e_i + e_j = e_k + e_l for multiple pairs, we can cancel
    
    # Try specific known constructions
    # p(x) = x^0 - x^1 + x^3 - x^4 + x^6 - x^7 + x^9 - x^10 + x^12 - x^13 + x^15 - x^16
    # This has period-3 structure in exponents: 0,1,3,4,6,7,9,10,12,13,15,16
    test_polys = [
        [(0,1),(1,-1),(3,1),(4,-1),(6,1),(7,-1),(9,1),(10,-1),(12,1),(13,-1),(15,1),(16,-1)],
        [(0,1),(1,-1),(3,-1),(4,1),(6,1),(7,-1),(9,-1),(10,1),(12,1),(13,-1),(15,-1),(16,1)],
        [(0,1),(2,1),(4,1),(6,1),(8,1),(10,1),(12,1),(14,1),(16,1),(18,1),(20,1),(22,1)],
        [(0,1),(2,-1),(4,1),(6,-1),(8,1),(10,-1),(12,1),(14,-1),(16,1),(18,-1),(20,1),(22,-1)],
    ]
    for tp in test_polys:
        score = evaluate(tp)
        if score < best_score:
            best_score = score
            best_poly = tp[:]
    
    if best_poly is None:
        # Fallback
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    return poly_to_list(best_poly)

