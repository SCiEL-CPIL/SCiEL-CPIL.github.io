import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Seed: 12 terms, all coefficient 1, exponents 0..11
    # Square has 23 nonzero terms
    best_poly = [0] * 12
    for i in range(12):
        best_poly[i] = 1
    best_nnz = 23  # (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count(exponents, coeffs):
        """Compute number of nonzero coefficients in square of polynomial
        given by exponents and coeffs lists (same length)."""
        n = len(exponents)
        sq = {}
        for i in range(n):
            ei, ci = exponents[i], coeffs[i]
            if ci == 0:
                continue
            for j in range(i, n):
                ej, cj = exponents[j], coeffs[j]
                if cj == 0:
                    continue
                key = ei + ej
                if i == j:
                    sq[key] = sq.get(key, 0) + ci * ci
                else:
                    sq[key] = sq.get(key, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    def build_poly(exponents, coeffs):
        """Build coefficient list from sorted exponents and coeffs."""
        max_exp = max(exponents) if exponents else 0
        result = [0] * (max_exp + 1)
        for e, c in zip(exponents, coeffs):
            result[e] = c
        return result
    
    random.seed(12345)
    
    # Strategy 1: Structured candidates
    # Try AP with various steps and sign patterns
    for step in range(1, 4):
        for start in range(0, 3):
            exponents = [start + i * step for i in range(12)]
            if max(exponents) > 40:
                continue
            sign_patterns = [
                [1]*12,
                [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
            ]
            for sp in sign_patterns:
                nnz = square_nonzero_count(exponents, sp)
                if nnz < best_nnz:
                    best_nnz = nnz
                    best_poly = build_poly(exponents, sp)
    
    # Strategy 2: Two groups with potential cancellation
    for gap in range(6, 25):
        exponents = list(range(0, 6)) + list(range(gap, gap + 6))
        if max(exponents) > 40:
            continue
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                coeffs = [s1]*6 + [s2]*6
                nnz = square_nonzero_count(exponents, coeffs)
                if nnz < best_nnz:
                    best_nnz = nnz
                    best_poly = build_poly(exponents, coeffs)
    
    # Strategy 3: Random search
    while time.time() < deadline - 5:
        # Random 12 distinct exponents in [0, 40]
        exponents = sorted(random.sample(range(0, 41), 12))
        
        # Random coefficients
        r = random.random()
        if r < 0.7:
            coeffs = [random.choice([1, -1]) for _ in range(12)]
        elif r < 0.9:
            coeffs = [random.choice([1, -1, 2, -2]) for _ in range(12)]
        else:
            coeffs = [random.choice([1, -1, 2, -2, 3, -3]) for _ in range(12)]
        
        nnz = square_nonzero_count(exponents, coeffs)
        
        if nnz < best_nnz:
            best_nnz = nnz
            best_poly = build_poly(exponents, coeffs)
            
            # If we found something good, do local search
            if best_nnz <= 15:
                for _ in range(500):
                    if time.time() > deadline - 5:
                        break
                    new_exponents = list(exponents)
                    new_coeffs = list(coeffs)
                    
                    # Perturb: change one coefficient
                    idx = random.randint(0, 11)
                    new_coeffs[idx] = random.choice([1, -1, 2, -2, -new_coeffs[idx]])
                    
                    nnz2 = square_nonzero_count(new_exponents, new_coeffs)
                    if nnz2 < best_nnz:
                        best_nnz = nnz2
                        best_poly = build_poly(new_exponents, new_coeffs)
                        exponents = new_exponents
                        coeffs = new_coeffs
    
    # Validate and return
    # Trim trailing zeros
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    # Count nonzero terms
    nz_count = sum(1 for c in best_poly if c != 0)
    
    # If invalid, fall back to seed
    if nz_count != 12 or best_poly[-1] == 0:
        best_poly = [1] * 12
    
    return best_poly
