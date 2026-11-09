"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product, combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square computation using FFT convolution."""
        n = len(coeff_list)
        if n == 0:
            return 0
        # Find max nonzero index to limit work
        max_idx = 0
        for i, c in enumerate(coeff_list):
            if c != 0:
                max_idx = i
        n_eff = max_idx + 1
        size = 1
        while size < 2 * n_eff - 1:
            size *= 2
        a = np.zeros(size, dtype=np.float64)
        for i in range(n_eff):
            if coeff_list[i] != 0:
                a[i] = coeff_list[i]
        fa = np.fft.fft(a)
        sq = np.fft.ifft(fa * fa).real
        sq = np.round(sq).astype(np.int64)
        sq = sq[:2 * n_eff - 1]
        return int(np.count_nonzero(sq))
    
    def square_nonzero_count_exact(coeffs):
        """Exact square computation using (exp, coeff) pairs."""
        n = len(coeffs)
        if n == 0:
            return 0
        sq = {}
        for i in range(n):
            ei, ci = coeffs[i]
            for j in range(i, n):
                ej, cj = coeffs[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    def validate_and_trim(poly):
        """Ensure poly meets constraints, return trimmed list or None."""
        if not poly:
            return None
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        if not poly:
            return None
        if len(poly) > 41:
            return None
        if poly[-1] == 0:
            return None
        nonzero_count = sum(1 for c in poly if c != 0)
        if nonzero_count != 12:
            return None
        if any(abs(c) > 64 for c in poly):
            return None
        return poly
    
    def try_poly(poly):
        """Try a polynomial, update best if valid and better."""
        nonlocal best_poly, best_score
        trimmed = validate_and_trim(poly[:])
        if trimmed is None:
            return
        score = square_nonzero_count_fast(trimmed)
        if score < best_score:
            best_score = score
            best_poly = trimmed
            if best_score <= 3:
                # Very good, can stop early in some phases
                pass
    
    # Phase 1: Basis polynomial superposition
    # Model P = w1*B1 + w2*B2 + w3*B3 where Bi are low-sparsity basis polynomials
    # Try trinomials as basis: B = a*x^p + b*x^q + c*x^r
    # Their squares have at most 6 terms (3 diagonal + 3 cross)
    # Superposition can create cancellations
    
    # Precompute some good trinomial bases
    # A trinomial x^a + x^b + x^c has square with terms at:
    # 2a, a+b, a+c, 2b, b+c, 2c (6 terms if all distinct)
    # With signs: (x^a - x^b + x^c)^2 can have cancellations
    
    # Try basis sets of trinomials and search over integer weights
    basis_trinomials = []
    # Generate trinomials with exponents in [0, 40]
    # Focus on structured ones
    for a in range(0, 15):
        for b in range(a+1, min(a+20, 30)):
            for c in range(b+1, min(b+15, 41)):
                if c > 40:
                    break
                # Only keep a subset to limit search
                if (a + b + c) % 3 == 0:  # Filter to reduce count
                    basis_trinomials.append((a, b, c))
    
    # Limit to a manageable number of basis polynomials
    random.seed(123)
    if len(basis_trinomials) > 200:
        basis_trinomials = random.sample(basis_trinomials, 200)
    
    # For each pair of basis trinomials, try linear combinations
    # P = w1*(x^a1 + x^b1 + x^c1) + w2*(x^a2 + x^b2 + x^c2)
    # This gives up to 6 terms, need to extend to 12
    # Better: use 4 basis polynomials of 3 terms each = 12 terms
    # Or 3 basis polynomials with varying weights to get 12 nonzero terms
    
    # Strategy: Pick 4 trinomial bases, assign weights to make exactly 12 nonzero
    # Actually, simpler: directly search for 12-term polynomials using structured families
    
    # Phase 1a: AP exponents with ±1 coefficients (quick)
    for d in range(1, 5):
        if time.time() > deadline:
            break
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    break
                poly = [0] * (exponents[-1] + 1)
                for i in range(12):
                    c = 1 if (signs_mask >> i) & 1 else -1
                    poly[exponents[i]] = c
                try_poly(poly)
    
    # Phase 1b: Basis superposition - combine 2-3 trinomials
    # Pick pairs of trinomials and try weight combinations
    # B1 = x^a + x^b + x^c, B2 = x^d + x^e + x^f
    # P = w1*B1 + w2*B2 (6 terms, need 12)
    # Instead: P = w1*B1 + w2*B2 + w3*B3 + w4*B4 (12 terms)
    # Or: use trinomials with overlapping exponents to create cancellations in square
    
    # Better approach: directly construct 12-term polynomials from structured patterns
    # that are known to have sparse squares
    
    # Known good patterns:
    # (1 + x + x^2 - x^3 - x^4 - x^5)^2 has fewer terms due to cancellation
    # Let's try polynomials that are products or sums of small sparse polynomials
    
    # Phase 2: Structured families with coefficient variations
    # Try polynomials of form: sum of c_i * x^(a_i) where a_i have structure
    
    # Family: pairs of terms that cancel in square
    # (x^a + x^b)^2 = x^(2a) + 2x^(a+b) + x^(2b)
    # (x^a - x^b)^2 = x^(2a) - 2x^(a+b) + x^(2b)
    # If we have both (x^a + x^b) and (x^c + x^d) with a+b = c+d,
    # the cross terms can cancel
    
    # Try: P = (x^a + x^b) + (x^c + x^d) + ... with a+b = c+d = ...
    # This creates cancellation in the square's middle terms
    
    # Phase 2a: Try polynomials with paired exponents
    # P = sum_{k=0}^{5} c_k * (x^(p_k) + x^(q_k)) where p_k + q_k = constant
    # This means all cross terms x^(p_k + q_j) for k!=j have the same exponent
    # if p_k + q_j = p_j + q_k, which holds if p_k - q_k = p_j - q_j
    
    # Let's try: P = sum c_k * x^(s + k*d) + d_k * x^(t - k*d)
    # where s + t = constant (palindromic structure)
    
    # Phase 2b: Random search with local improvement
    random.seed(42)
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Generate 12 random distinct exponents
        exponents = sorted(random.sample(range(0, 41), 12))
        
        # Try several coefficient patterns
        coeff_patterns = [
            [1]*12,
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
            [1,1,1,1,1,1,1,1,1,1,1,-1],
            [1,1,-1,1,1,-1,1,1,-1,1,1,-1],
            [1,1,1,-1,1,1,1,-1,1,1,1,-1],
        ]
        
        for coeff_pattern in coeff_patterns:
            poly = [0] * (exponents[-1] + 1)
            for i in range(12):
                poly[exponents[i]] = coeff_pattern[i]
            try_poly(poly)
        
        # Also try with some ±2 coefficients
        for _ in range(3):
            coeff_pattern = [random.choice([-2,-1,1,2]) for _ in range(12)]
            poly = [0] * (exponents[-1] + 1)
            for i in range(12):
                poly[exponents[i]] = coeff_pattern[i]
            try_poly(poly)
    
    # Phase 3: Local search - perturb best solution
    if best_poly is not None and time.time() < deadline:
        work_poly = best_poly[:]
        work_count = best_score
        max_degree = len(work_poly)
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            # Randomly pick a position to modify
            pos = random.randint(0, max_degree - 1)
            old_val = work_poly[pos]
            
            # Try different values
            new_vals = []
            if old_val == 0:
                new_vals = [1, -1, 2, -2]
            elif abs(old_val) <= 3:
                new_vals = [v for v in range(-3, 4) if v != old_val and abs(v) <= 4]
            else:
                new_vals = [1, -1, 2, -2, 0]
            
            for new_val in new_vals:
                if abs(new_val) > 64:
                    continue
                
                work_poly[pos] = new_val
                
                # Check if still exactly 12 nonzero
                nonzero = sum(1 for c in work_poly if c != 0)
                if nonzero != 12:
                    work_poly[pos] = old_val
                    continue
                
                # Ensure final coefficient is nonzero
                if work_poly[-1] == 0:
                    work_poly[pos] = old_val
                    continue
                
                cnt = square_nonzero_count_fast(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_score:
                        best_score = cnt
                        best_poly = validate_and_trim(work_poly[:])
                else:
                    work_poly[pos] = old_val
    
    # Phase 4: Try specific algebraic structures
    # (1+x)^12 has 13 terms, not 12
    # (1+x+x^2)^4 has more terms
    # Try: (1+x)^6 * (1+x^2)^2 = (1+6x+15x^2+20x^3+15x^4+6x^5+x^6)*(1+2x^2+x^4)
    # This is 13 terms, need to zero out one
    
    # Try polynomials that are near-perfect squares
    # If P = Q + R where Q is a square and R is small, P^2 = Q^2 + 2QR + R^2
    # The 2QR terms can create cancellations if R is chosen well
    
    # Phase 5: Exhaustive search over small coefficient sets for specific exponent structures
    # Try exponents that form a group under addition (mod something)
    
    # Try: exponents = {0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16}
    # This is pairs (0,1), (3,4), (6,7), (9,10), (12,13), (15,16)
    # Differences: 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1
    # Sum structure: many pair sums collide
    
    structured_exponents = [
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22],
    ]
    
    for exp_set in structured_exponents:
        if time.time() > deadline:
            break
        # Try all sign patterns (fix first = +1)
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline:
                break
            poly = [0] * (exp_set[-1] + 1)
            poly[exp_set[0]] = 1
            for i, s in enumerate(signs):
                poly[exp_set[i+1]] = s
            try_poly(poly)
    
    # Phase 6: Try with coefficient magnitudes
    for exp_set in structured_exponents[:4]:
        if time.time() > deadline:
            break
        # Try some coefficients as ±2
        for n_twos in range(2, 5):
            if time.time() > deadline:
                break
            for two_positions in combinations(range(12), n_twos):
                if time.time() > deadline:
                    break
                # Try a few sign patterns
                for signs in list(product([1, -1], repeat=11))[:500]:
                    if time.time() > deadline:
                        break
                    poly = [0] * (exp_set[-1] + 1)
                    poly[exp_set[0]] = 1
                    for i, s in enumerate(signs):
                        val = s
                        if (i+1) in two_positions:
                            val *= 2
                        poly[exp_set[i+1]] = val
                    try_poly(poly)
    
    # Return best found, or a valid fallback
    if best_poly is not None:
        return best_poly
    
    # Fallback: 12-term polynomial
    return [1]*12

# EVOLVE-BLOCK-END
