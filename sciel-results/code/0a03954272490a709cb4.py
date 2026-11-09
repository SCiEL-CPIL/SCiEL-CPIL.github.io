import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin for 90s limit
    
    def count_nonzero(poly):
        return sum(1 for c in poly if c != 0)
    
    def poly_square(coeffs):
        """Compute exact square of polynomial given as list of coefficients (ascending degree)."""
        n = len(coeffs)
        result = [0] * (2 * n - 1)
        nonzero_indices = [i for i, c in enumerate(coeffs) if c != 0]
        for ii in range(len(nonzero_indices)):
            i = nonzero_indices[ii]
            ci = coeffs[i]
            for jj in range(ii, len(nonzero_indices)):
                j = nonzero_indices[jj]
                cj = coeffs[j]
                val = ci * cj
                idx = i + j
                if ii == jj:
                    result[idx] += val
                else:
                    result[idx] += 2 * val
        return result
    
    def score_poly(coeffs):
        """Return number of nonzero terms in the square."""
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    def valid(coeffs):
        """Check if polynomial meets all constraints."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nz = count_nonzero(coeffs)
        if nz != 12:
            return False
        for c in coeffs:
            if not (-64 <= c <= 64):
                return False
        return True
    
    # Try some structured constructions first
    best_poly = None
    best_score = float('inf')
    
    # Construction 1: Try to make P a scaled version of a known sparse square root
    # If P(x) = c * (x^a + x^b + ...), P^2 = c^2 * (sum of pair terms)
    # We want the pair sums to collide
    
    # Try exponents that form a "Sidon-like" set but with collisions
    # For 12 terms, if exponents are {0, 1, 2, ..., 11}, sums are 0..22 (23 terms)
    # If exponents are {0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22} - structured
    
    # Let's try: exponents in pairs to create cancellation
    # P(x) = a0 + a1*x + ... with specific coefficients
    
    # Random search with smart structure
    random.seed(42)
    
    # Try structured exponent sets
    exponent_sets = [
        list(range(12)),  # 0..11
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # even exponents
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
    ]
    
    for exps in exponent_sets:
        if len(set(exps)) != 12:
            continue
        if max(exps) >= 41:
            continue
        # Try different coefficient assignments
        for _ in range(200):
            coeffs = [0] * (max(exps) + 1)
            for e in exps:
                coeffs[e] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -8, 8, -16, 16, -32, 32])
            if valid(coeffs):
                s = score_poly(coeffs)
                if s < best_score:
                    best_score = s
                    best_poly = coeffs[:]
    
    # Random search
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        # Random degree
        max_deg = random.randint(11, 40)
        # Choose 12 distinct exponents
        exps = sorted(random.sample(range(max_deg + 1), 12))
        coeffs = [0] * (max_deg + 1)
        for e in exps:
            coeffs[e] = random.randint(-64, 64)
            if coeffs[e] == 0:
                coeffs[e] = random.choice([-1, 1])
        
        if valid(coeffs):
            s = score_poly(coeffs)
            if s < best_score:
                best_score = s
                best_poly = coeffs[:]
        
        # Every 1000 iterations, try a more structured approach
        if iterations % 1000 == 0:
            # Try symmetric polynomials: P(x) = sum a_i * x^(i + k) for some structure
            base = random.randint(0, 10)
            for _ in range(50):
                coeffs2 = [0] * (base + 41)
                if len(coeffs2) > 41:
                    coeffs2 = [0] * 41
                # Place 12 terms
                positions = sorted(random.sample(range(len(coeffs2)), 12))
                for p in positions:
                    coeffs2[p] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4])
                if valid(coeffs2):
                    s = score_poly(coeffs2)
                    if s < best_score:
                        best_score = s
                        best_poly = coeffs2[:]
    
    if best_poly is None:
        # Fallback: a valid polynomial
        best_poly = [0] * 12
        for i in range(12):
            best_poly[i] = 1
        best_poly[11] = 1  # ensure nonzero final
    
    # Ensure final coefficient is nonzero
    while best_poly[-1] == 0:
        best_poly.pop()
    
    return best_poly
