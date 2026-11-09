"""Rewritten program: basis-combination approach with simulated annealing refinement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    def square_nonzero_count(coeff_list):
        """Exact square computation using convolution of nonzero terms."""
        n = len(coeff_list)
        if n == 0:
            return 0
        nz = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
        if not nz:
            return 0
        max_deg = 2 * (len(coeff_list) - 1)
        sq = [0] * (max_deg + 1)
        for i, ci in nz:
            for j, cj in nz:
                sq[i + j] += ci * cj
        return sum(1 for v in sq if v != 0)
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square using FFT for longer polynomials."""
        n = len(coeff_list)
        if n == 0:
            return 0
        size = 1
        while size < 2 * n - 1:
            size *= 2
        a = np.zeros(size, dtype=np.float64)
        for i, c in enumerate(coeff_list):
            if c != 0:
                a[i] = c
        fa = np.fft.fft(a)
        sq = np.fft.ifft(fa * fa).real
        sq = np.round(sq).astype(np.int64)
        sq = sq[:2 * n - 1]
        return int(np.count_nonzero(sq))
    
    def trim(poly):
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def is_valid(poly):
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 or c != int(c) for c in poly):
            return False
        return True
    
    # Seed: (1+x+...+x^11)^2 has 23 nonzero terms
    best_poly = [1] * 12
    best_count = 23
    
    # ============================================================
    # STRATEGY: Build polynomials as linear combinations of 2-3
    # basis polynomials with known sparse squares, then search
    # over integer weights and exponent shifts for cancellations.
    # ============================================================
    
    def build_from_basis(basis_polys, weights, shifts):
        """Build a polynomial as sum of weighted, shifted basis polynomials.
        Returns (coeff_list, valid) where valid checks 12 nonzero constraint."""
        max_deg = 0
        for bp, s in zip(basis_polys, shifts):
            max_deg = max(max_deg, len(bp) - 1 + s)
        
        if max_deg >= 41:
            return None, False
        
        result = [0] * (max_deg + 1)
        for bp, w, s in zip(basis_polys, weights, shifts):
            for i, c in enumerate(bp):
                if c != 0:
                    result[i + s] += w * c
        
        # Trim
        result = trim(result)
        
        # Check exactly 12 nonzero
        nz = sum(1 for c in result if c != 0)
        if nz != 12:
            return None, False
        
        # Check coefficient bounds
        if any(abs(c) > 64 for c in result):
            return None, False
        
        # Check final coefficient nonzero
        if result[-1] == 0:
            return None, False
        
        return result, True
    
    # Define basis polynomial templates
    # Each basis is (coeff_list, description)
    # We'll use short polynomials: trinomials, quadrinomials, etc.
    
    basis_templates = [
        # Trinomials: a + b*x^p + c*x^q
        ([1, 0, 0, 1], "1+x^3"),
        ([1, 0, 0, -1], "1-x^3"),
        ([1, 0, 0, 0, 1], "1+x^4"),
        ([1, 0, 0, 0, -1], "1-x^4"),
        ([1, 0, 0, 0, 0, 1], "1+x^5"),
        ([1, 0, 0, 0, 0, -1], "1-x^5"),
        ([1, 0, 0, 1, 0, 0, 1], "1+x^3+x^6"),
        ([1, 0, 0, 1, 0, 0, -1], "1+x^3-x^6"),
        # Quadrinomials
        ([1, 1, 0, 1, 1], "1+x+x^3+x^4"),
        ([1, -1, 0, 1, -1], "1-x+x^3-x^4"),
        ([1, 0, 1, 0, 1], "1+x^2+x^4"),
        ([1, 0, -1, 0, 1], "1-x^2+x^4"),
        # 4-term
        ([1, 0, 1, 0, 1, 0, 1], "1+x^2+x^4+x^6"),
        ([1, 0, -1, 0, 1, 0, -1], "1-x^2+x^4-x^6"),
        # 5-term
        ([1, 0, 0, 1, 0, 0, 1, 0, 0, 1], "1+x^3+x^6+x^9"),
        ([1, 0, 0, -1, 0, 0, 1, 0, 0, -1], "1-x^3+x^6-x^9"),
        # Consecutive blocks
        ([1, 1, 1], "1+x+x^2"),
        ([1, 1, 1, 1], "1+x+x^2+x^3"),
        ([1, 1, 1, 1, 1], "1+x+x^2+x^3+x^4"),
        ([1, -1, 1, -1], "1-x+x^2-x^3"),
        ([1, -1, 1, -1, 1], "1-x+x^2-x^3+x^4"),
    ]
    
    # ============================================================
    # Phase 1: Systematic search over 2-basis combinations
    # ============================================================
    
    def search_two_basis():
        nonlocal best_poly, best_count
        
        n_templates = len(basis_templates)
        
        for i in range(n_templates):
            if time.time() > deadline:
                return
            bp1 = basis_templates[i][0]
            
            for j in range(i, n_templates):
                if time.time() > deadline:
                    return
                bp2 = basis_templates[j][0]
                
                # Try weights and shifts
                # Total nonzero terms must be 12
                # If bp1 has n1 nonzero and bp2 has n2 nonzero,
                # and they don't overlap, total = n1 + n2 (with weights)
                # But overlaps can create cancellations or extra nonzeros
                
                n1 = sum(1 for c in bp1 if c != 0)
                n2 = sum(1 for c in bp2 if c != 0)
                
                # We need total nonzero = 12 after combining
                # With weights w1, w2: terms can cancel if same position has opposite signs
                
                # Try shifts to separate or overlap
                max_shift = 40 - max(len(bp1), len(bp2)) + 1
                if max_shift < 0:
                    continue
                
                # Try a subset of shifts and weights
                weight_range = range(-3, 4)
                
                for s1 in range(0, min(5, max_shift + 1)):
                    if time.time() > deadline:
                        return
                    for s2 in range(s1, min(s1 + max_shift, 41 - len(bp2))):
                        if time.time() > deadline:
                            return
                        for w1 in weight_range:
                            if w1 == 0:
                                continue
                            for w2 in weight_range:
                                if w2 == 0:
                                    continue
                                
                                poly, valid = build_from_basis(
                                    [bp1, bp2], [w1, w2], [s1, s2]
                                )
                                if valid:
                                    cnt = square_nonzero_count(poly)
                                    if cnt < best_count:
                                        best_count = cnt
                                        best_poly = poly[:]
    
    # ============================================================
    # Phase 2: Systematic search over 3-basis combinations
    # ============================================================
    
    def search_three_basis():
        nonlocal best_poly, best_count
        
        # Use a smaller set of basis templates for 3-way combos
        small_templates = basis_templates[:10]
        n = len(small_templates)
        
        for i in range(n):
            if time.time() > deadline:
                return
            bp1 = small_templates[i][0]
            
            for j in range(i, n):
                if time.time() > deadline:
                    return
                bp2 = small_templates[j][0]
                
                for k in range(j, n):
                    if time.time() > deadline:
                        return
                    bp3 = small_templates[k][0]
                    
                    n1 = sum(1 for c in bp1 if c != 0)
                    n2 = sum(1 for c in bp2 if c != 0)
                    n3 = sum(1 for c in bp3 if c != 0)
                    
                    # Try limited shifts
                    max_s1 = 3
                    max_s2 = 3
                    max_s3 = 3
                    
                    for s1 in range(max_s1 + 1):
                        if time.time() > deadline:
                            return
                        for s2 in range(max_s2 + 1):
                            if time.time() > deadline:
                                return
                            for s3 in range(max_s3 + 1):
                                if time.time() > deadline:
                                    return
                                for w1 in [-2, -1, 1, 2]:
                                    for w2 in [-2, -1, 1, 2]:
                                        for w3 in [-2, -1, 1, 2]:
                                            poly, valid = build_from_basis(
                                                [bp1, bp2, bp3],
                                                [w1, w2, w3],
                                                [s1, s2, s3]
                                            )
                                            if valid:
                                                cnt = square_nonzero_count(poly)
                                                if cnt < best_count:
                                                    best_count = cnt
                                                    best_poly = poly[:]
    
    # ============================================================
    # Phase 3: Structured exponent search with sign patterns
    # ============================================================
    
    def search_structured_exponents():
        nonlocal best_poly, best_count
        
        # Key exponent families designed for sum-collisions
        # If many pairs (i,j) give the same i+j, coefficients can cancel
        
        families = [
            # Consecutive (seed)
            list(range(12)),
            # Pairs: (0,1),(3,4),(6,7),(9,10),(12,13)
            [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
            # Triplets: (0,1,2),(4,5,6),(8,9,10)
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13],
            # Symmetric pairs
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            # Arithmetic progression step 2
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            # Two clusters
            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
            # Three clusters
            [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15],
            # Alternating dense/sparse
            [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
            # Power-of-2 based
            [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36],
            # Geometric-ish
            [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
            # Palindromic structure
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            # Anti-symmetric potential
            [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15],
            # Random-ish structured
            [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        ]
        
        for exp_set in families:
            if time.time() > deadline:
                return
            if max(exp_set) >= 41:
                continue
            
            # Try sign patterns: fix first as +1
            # For 11 remaining positions, try structured sign patterns
            # rather than all 2^11
            
            # Pattern 1: all +1
            coeffs = [0] * 41
            for e in exp_set:
                coeffs[e] = 1
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
            
            # Pattern 2: alternating
            coeffs = [0] * 41
            for idx, e in enumerate(exp_set):
                coeffs[e] = 1 if idx % 2 == 0 else -1
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
            
            # Pattern 3: first half +, second half -
            coeffs = [0] * 41
            for idx, e in enumerate(exp_set):
                coeffs[e] = 1 if idx < 6 else -1
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
            
            # Pattern 4: symmetric +
            coeffs = [0] * 41
            for idx, e in enumerate(exp_set):
                coeffs[e] = 1
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
            
            # Random sign patterns
            random.seed(hash(tuple(exp_set)) % (2**31))
            for _ in range(50):
                if time.time() > deadline:
                    return
                coeffs = [0] * 41
                coeffs[exp_set[0]] = 1
                for e in exp_set[1:]:
                    coeffs[e] = random.choice([-1, 1])
                cnt = square_nonzero_count_fast(coeffs)
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(coeffs[:])
    
    # ============================================================
    # Phase 4: Simulated annealing over exponent+coefficient space
    # ============================================================
    
    def simulated_annealing():
        nonlocal best_poly, best_count
        
        random.seed(42)
        np.random.seed(42)
        
        # Start from best found so far
        current = best_poly[:]
        while len(current) < 41:
            current.append(0)
        current_count = best_count
        
        # Get current nonzero positions and values
        def get_nz(poly):
            return [(i, c) for i, c in enumerate(poly) if c != 0]
        
        def set_from_nz(nz_list):
            poly = [0] * 41
            for i, c in nz_list:
                if 0 <= i < 41:
                    poly[i] = c
            return poly
        
        # Initial state
        nz = get_nz(current)
        if len(nz) != 12:
            nz = [(i, 1) for i in range(12)]
        
        temp = 5.0
        cooling = 0.997
        min_temp = 0.1
        max_iters = 30000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if temp < min_temp:
                break
            
            # Generate neighbor
            nz_new = [(i, c) for i, c in nz]
            op = random.random()
            
            if op < 0.4:
                # Move one term
                idx = random.randint(0, 11)
                old_pos, old_coeff = nz_new[idx]
                new_pos = old_pos + random.randint(-3, 3)
                new_coeff = old_coeff + random.randint(-2, 2)
                if new_coeff == 0:
                    new_coeff = old_coeff + random.choice([-1, 1])
                if abs(new_coeff) > 64:
                    new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
                if 0 <= new_pos < 41:
                    nz_new[idx] = (new_pos, new_coeff)
                    # Check for duplicate positions
                    positions = [p for p, c in nz_new]
                    if len(set(positions)) == 12:
                        pass
                    else:
                        nz_new = [(i, c) for i, c in nz]
                        continue
            elif op < 0.6:
                # Change coefficient of one term
                idx = random.randint(0, 11)
                old_pos, old_coeff = nz_new[idx]
                new_coeff = old_coeff + random.choice([-2, -1, 1, 2])
                if new_coeff == 0:
                    new_coeff = random.choice([-1, 1])
                if abs(new_coeff) > 64:
                    new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
                nz_new[idx] = (old_pos, new_coeff)
            elif op < 0.75:
                # Swap two terms
                i, j = random.sample(range(12), 2)
                nz_new[i], nz_new[j] = nz_new[j], nz_new[i]
            elif op < 0.9:
                # Move one term to random position
                idx = random.randint(0, 11)
                old_pos, old_coeff = nz_new[idx]
                new_pos = random.randint(0, 40)
                if new_pos != old_pos:
                    nz_new[idx] = (new_pos, old_coeff)
                    positions = [p for p, c in nz_new]
                    if len(set(positions)) != 12:
                        nz_new = [(i, c) for i, c in nz]
                        continue
            else:
                # Change two coefficients
                i, j = random.sample(range(12), 2)
                pi, ci = nz_new[i]
                pj, cj = nz_new[j]
                ci_new = ci + random.choice([-1, 1, 2, -2])
                cj_new = cj + random.choice([-1, 1, 2, -2])
                if ci_new == 0:
                    ci_new = ci + random.choice([-1, 1])
                if cj_new == 0:
                    cj_new = cj + random.choice([-1, 1])
                if abs(ci_new) <= 64 and abs(cj_new) <= 64:
                    nz_new[i] = (pi, ci_new)
                    nz_new[j] = (pj, cj_new)
            
            # Build polynomial and evaluate
            poly_new = set_from_nz(nz_new)
            poly_new = trim(poly_new)
            
            if len(poly_new) < 1 or poly_new[-1] == 0:
                continue
            
            nz_count = sum(1 for c in poly_new if c != 0)
            if nz_count != 12:
                continue
            
            if any(abs(c) > 64 for c in poly_new):
                continue
            
            cnt_new = square_nonzero_count(poly_new)
            delta = cnt_new - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current = poly_new
                current_count = cnt_new
                nz = nz_new[:]
                
                if cnt_new < best_count:
                    best_count = cnt_new
                    best_poly = trim(poly_new[:])
            
            temp *= cooling
    
    # ============================================================
    # Phase 5: Palindromic and anti-palindromic structured search
    # ============================================================
    
    def search_palindromic():
        nonlocal best_poly, best_count
        
        # For a palindromic polynomial of degree d: p(x) = x^d * p(1/x)
        # The square will also be palindromic
        # For anti-palindromic: p(x) = -x^d * p(1/x)
        
        # Try palindromic 12-term polynomials
        # Degree must be even for 12 terms to be palindromic
        # p(x) = a0 + a1*x + ... + a6*x^6 + a6*x^7 + ... + a0*x^12 (degree 12, 13 terms - no)
        # Actually for 12 terms: degree 11 with p(x) = a0 + ... + a11*x^11
        # Palindromic: a_i = a_{11-i}, so 6 independent coefficients, 12 terms total
        
        for d in [11, 13, 15, 17, 19, 21, 23]:
            if d + 1 > 41:
                continue
            if time.time() > deadline:
                return
            
            # For degree d with 12 nonzero terms in palindromic structure
            # The middle terms (if d is odd) are self-paired
            # Number of independent coefficients: (d+1)/2 if d odd, (d+2)/2 if d even
            # But we need exactly 12 nonzero total
            
            # Simpler: just try specific structured patterns
            # Anti-palindromic of degree 11: a_i = -a_{11-i}
            # This gives 6 independent values, and the middle (x^5) must be 0
            # So 11 terms max from anti-palindromic of degree 11... not quite 12
            
            # Let's try: 12-term polynomial where coefficients come in ±pairs
            # Positions: (0,11), (1,10), (2,9), (3,8), (4,7), (5,6)
            # Anti-palindromic: c[i] = -c[11-i]
            
            random.seed(99 + d)
            for _ in range(100):
                if time.time() > deadline:
                    return
                coeffs = [0] * (d + 1)
                for i in range(6):
                    val = random.choice([-3, -2, -1, 1, 2, 3])
                    coeffs[i] = val
                    coeffs[d - i] = -val  # anti-palindromic
                # This gives 12 nonzero terms (assuming all 6 values are nonzero)
                nz = sum(1 for c in coeffs if c != 0)
                if nz == 12 and coeffs[-1] != 0:
                    cnt = square_nonzero_count_fast(coeffs)
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(coeffs[:])
            
            # Palindromic: c[i] = c[d-i]
            for _ in range(100):
                if time.time() > deadline:
                    return
                coeffs = [0] * (d + 1)
                for i in range(6):
                    val = random.choice([-3, -2, -1, 1, 2, 3])
                    coeffs[i] = val
                    coeffs[d - i] = val
                nz = sum(1 for c in coeffs if c != 0)
                if nz == 12 and coeffs[-1] != 0:
                    cnt = square_nonzero_count_fast(coeffs)
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(coeffs[:])
    
    # ============================================================
    # Phase 6: Local refinement on best solution
    # ============================================================
    
    def local_refinement():
        nonlocal best_poly, best_count
        
        random.seed(777)
        work = best_poly[:]
        while len(work) < 41:
            work.append(0)
        work_count = best_count
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a position to modify
            pos = random.randint(0, 40)
            old_val = work[pos]
            
            if old_val == 0:
                candidates = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
            else:
                candidates = [0, 1, -1, 2, -2, 3, -3, old_val + 1, old_val - 1]
            
            for new_val in candidates:
                if new_val == old_val or abs(new_val) > 64:
                    continue
                
                work[pos] = new_val
                nz = sum(1 for c in work if c != 0)
                if nz != 12 or work[-1] == 0:
                    work[pos] = old_val
                    continue
                
                cnt = square_nonzero_count_fast(work)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work[:])
                else:
                    work[pos] = old_val
    
    # ============================================================
    # Execute all phases
    # ============================================================
    
    # Phase 3 first (structured exponents) - fast and systematic
    search_structured_exponents()
    
    # Phase 1: Two-basis combinations
    if time.time() < deadline:
        search_two_basis()
    
    # Phase 2: Three-basis combinations (limited)
    if time.time() < deadline:
        search_three_basis()
    
    # Phase 5: Palindromic/anti-palindromic
    if time.time() < deadline:
        search_palindromic()
    
    # Phase 4: Simulated annealing
    if time.time() < deadline:
        simulated_annealing()
    
    # Phase 6: Local refinement
    if time.time() < deadline:
        local_refinement()
    
    # Final cleanup
    best_poly = trim(best_poly)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        return [1] * 12
    
    for i in range(len(best_poly)):
        best_poly[i] = int(best_poly[i])
        if best_poly[i] < -64:
            best_poly[i] = -64
        elif best_poly[i] > 64:
            best_poly[i] = 64
    
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
