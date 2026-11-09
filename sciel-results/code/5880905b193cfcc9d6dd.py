"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    best_poly = [1] * 12
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square computation using FFT convolution."""
        n = len(coeff_list)
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
    
    # Phase 1: Try known sparse-square constructions
    # Cyclotomic-based: Φ_n(x) for various n, then take products or sums
    cyclotomic_candidates = [
        # Φ_1(x) = x-1, Φ_2(x)=x+1, Φ_3(x)=x²+x+1, Φ_4(x)=x²+1, Φ_5(x)=x⁴+x³+x²+x+1
        # Try combinations that give exactly 12 terms
        [1, -1],  # Φ_1
        [1, 1],   # Φ_2
        [1, 1, 1],  # Φ_3
        [1, 0, 1],  # Φ_4
        [1, 1, 1, 1, 1],  # Φ_5
        [1, 0, 0, 0, 0, 1],  # Φ_6
        [1, 1, 0, 1, 1, 0, 1],  # Φ_7
        [1, 0, 1, 0, 1, 0, 1, 0, 1],  # Φ_9
    ]
    
    # Try products of small cyclotomics to get 12-term polynomials
    def poly_mul(a, b):
        """Multiply two polynomials (coefficient lists)."""
        result = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                result[i + j] += ai * bj
        return result
    
    def poly_add(a, b):
        """Add two polynomials."""
        max_len = max(len(a), len(b))
        result = [0] * max_len
        for i, ai in enumerate(a):
            result[i] += ai
        for i, bi in enumerate(b):
            result[i] += bi
        return result
    
    def poly_sub(a, b):
        """Subtract b from a."""
        max_len = max(len(a), len(b))
        result = [0] * max_len
        for i, ai in enumerate(a):
            result[i] += ai
        for i, bi in enumerate(b):
            result[i] -= bi
        return result
    
    # Try sums/differences of cyclotomic polynomials to get 12 terms
    for i in range(len(cyclotomic_candidates)):
        if time.time() > deadline:
            break
        for j in range(i, len(cyclotomic_candidates)):
            if time.time() > deadline:
                break
            # Try sum
            s = poly_add(cyclotomic_candidates[i], cyclotomic_candidates[j])
            s = trim(s)
            nonzero = sum(1 for c in s if c != 0)
            if nonzero == 12 and is_valid(s):
                cnt = square_nonzero_count_fast(s)
                if cnt < best_count:
                    best_count = cnt
                    best_poly = s[:]
            
            # Try difference
            if i != j:
                d = poly_sub(cyclotomic_candidates[i], cyclotomic_candidates[j])
                d = trim(d)
                nonzero = sum(1 for c in d if c != 0)
                if nonzero == 12 and is_valid(d):
                    cnt = square_nonzero_count_fast(d)
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = d[:]
    
    # Phase 2: Golay-like sequences with structured signs
    # Golay complementary pairs have minimal autocorrelation
    golay_bases = [
        [1, 1, 1, 1, -1, -1, 1, -1],
        [1, 1, 1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, 1, -1, -1, 1],
    ]
    
    # Extend Golay sequences to 12 terms by padding or combining
    for base in golay_bases:
        if time.time() > deadline:
            break
        # Try different extensions
        for pad_left in range(0, 5):
            for pad_right in range(0, 5 - pad_left + 1):
                if time.time() > deadline:
                    break
                # Try different fill values for padding
                for fill_left in [0, 1, -1]:
                    for fill_right in [0, 1, -1]:
                        if time.time() > deadline:
                            break
                        poly = [fill_left] * pad_left + base + [fill_right] * pad_right
                        # Trim to ensure we have meaningful polynomial
                        poly = trim(poly)
                        nonzero = sum(1 for c in poly if c != 0)
                        if nonzero != 12:
                            continue
                        if not is_valid(poly):
                            continue
                        cnt = square_nonzero_count_fast(poly)
                        if cnt < best_count:
                            best_count = cnt
                            best_poly = poly[:]
    
    # Phase 3: Systematic search over exponent sets with sign patterns
    # Focus on sets that create good cancellation patterns
    exponent_families = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 17],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
    ]
    
    for exp_set in exponent_families:
        if time.time() > deadline:
            break
        # Try all sign patterns (fix first = +1)
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline:
                break
            coeffs = [0] * 41
            coeffs[exp_set[0]] = 1
            for i, s in enumerate(signs):
                coeffs[exp_set[i+1]] = s
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
    
    # Phase 4: Local refinement with targeted moves
    if time.time() < deadline and is_valid(best_poly):
        work_poly = best_poly[:]
        work_count = best_count
        max_degree = len(work_poly)
        
        # Targeted local search: try changing 1-2 coefficients at a time
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            n_mods = 1 if np.random.random() < 0.8 else 2
            positions = np.random.choice(max_degree, size=n_mods, replace=False)
            original = [work_poly[p] for p in positions]
            
            new_vals = []
            for p in positions:
                choices = [1, -1, 2, -2, 3, -3, 0, 4, -4]
                new_vals.append(np.random.choice(choices))
            
            for p, nv in zip(positions, new_vals):
                work_poly[p] = nv
            
            nonzero = sum(1 for c in work_poly if c != 0)
            if nonzero != 12 or work_poly[-1] == 0:
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
                continue
            
            if any(abs(c) > 64 for c in work_poly):
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
                continue
            
            cnt = square_nonzero_count_fast(work_poly)
            if cnt < work_count:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(work_poly[:])
            else:
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
    
    # Final cleanup
    best_poly = trim(best_poly)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
