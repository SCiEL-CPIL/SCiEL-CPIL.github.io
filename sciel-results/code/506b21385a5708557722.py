"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    # Helper: compute sparse square coefficient count
    def square_nonzero_count(sparse_coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(sparse_coeffs)
        if n == 0:
            return 0
        sq = {}
        for i in range(n):
            ei, ci = sparse_coeffs[i]
            for j in range(i, n):
                ej, cj = sparse_coeffs[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    # Helper: validate and trim
    def validate_and_trim(poly):
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
    
    # Helper: build polynomial from sparse representation
    def build_poly(sparse_coeffs):
        if not sparse_coeffs:
            return None
        max_exp = max(e for e, c in sparse_coeffs)
        poly = [0] * (max_exp + 1)
        for exp, c in sparse_coeffs:
            poly[exp] = c
        return validate_and_trim(poly)
    
    # Helper: try a candidate and update best
    def try_candidate(sparse_coeffs):
        nonlocal best_poly, best_score
        score = square_nonzero_count(sparse_coeffs)
        if score < best_score:
            best_score = score
            poly = build_poly(sparse_coeffs)
            if poly is not None:
                best_poly = poly
        return score
    
    # Phase 1: AP with ±1 coefficients (fast exhaustive)
    for d in range(1, 4):
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    return [1]*12
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)
    
    # Phase 2: Product-based construction with Golay-like factors
    random.seed(42)
    coeff_choices = [-2, -1, 1, 2]
    
    # Pre-compute small sparse factors with good square properties
    good_factors = []
    for nterms in range(2, 7):
        for exps in __import__('itertools').combinations(range(0, 15), nterms):
            for signs_mask in range(1 << nterms):
                if time.time() > deadline:
                    break
                coeffs = [(exps[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(nterms)]
                sq_count = square_nonzero_count(coeffs)
                # Keep factors with relatively sparse squares
                if sq_count <= nterms + 2:
                    good_factors.append(coeffs)
    
    # Try products of two good factors
    for trial in range(800):
        if time.time() > deadline:
            break
        
        if len(good_factors) >= 2:
            f1 = random.choice(good_factors)
            f2 = random.choice(good_factors)
        else:
            continue
        
        # Compute product
        prod = {}
        for e1, c1 in f1:
            for e2, c2 in f2:
                e = e1 + e2
                prod[e] = prod.get(e, 0) + c1 * c2
        
        sparse = [(e, c) for e, c in sorted(prod.items()) if c != 0]
        
        if len(sparse) != 12:
            continue
        if sparse[-1][0] > 40:
            continue
        if any(abs(c) > 64 for _, c in sparse):
            continue
        
        try_candidate(sparse)
    
    # Phase 3: Random search with structured initialization
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        # Generate random sparse polynomial
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice(coeff_choices) for _ in range(12)]
        sparse = [(exps[i], coeffs[i]) for i in range(12)]
        try_candidate(sparse)
    
    # Phase 4: Local search / hill climbing from best found
    if best_poly is not None:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        
        for iteration in range(8000):
            if time.time() > deadline:
                break
            
            mutated = list(sparse_best)
            mutation_type = random.randint(0, 4)
            
            if mutation_type == 0 and len(mutated) > 2:
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 1:
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-4, 4)
                if 0 <= new_e <= 40:
                    if not any(ne == new_e for ne, nc in mutated if (ne, nc) != mutated[idx]):
                        mutated[idx] = (new_e, c)
            elif mutation_type == 2:
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    if ei != ej:
                        mutated[i] = (ej, ci)
                        mutated[j] = (ei, cj)
            elif mutation_type == 3:
                # Perturb two coefficients
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    new_ci = random.choice([-3, -2, -1, 1, 2, 3])
                    new_cj = random.choice([-3, -2, -1, 1, 2, 3])
                    mutated[i] = (ei, new_ci)
                    mutated[j] = (ej, new_cj)
            else:
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice(coeff_choices) for _ in range(12)]
                mutated = [(exps[i], coeffs[i]) for i in range(12)]
            
            if len(mutated) != 12:
                continue
            if any(e < 0 or e > 40 for e, c in mutated):
                continue
            if any(abs(c) > 64 for e, c in mutated):
                continue
            exps_list = [e for e, c in mutated]
            if len(set(exps_list)) != 12:
                continue
            
            score = square_nonzero_count(mutated)
            if score < best_score:
                best_score = score
                poly = build_poly(mutated)
                if poly is not None:
                    best_poly = poly
                    sparse_best = mutated
    
    # Phase 5: Palindromic structures
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        base_exps = list(range(12))
        shifted_exps = [e + d_shift for e in base_exps]
        if shifted_exps[-1] > 40:
            continue
        for c0 in [-2, -1, 1, 2]:
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    for c3 in [-2, -1, 1, 2]:
                        for c4 in [-2, -1, 1, 2]:
                            for c5 in [-2, -1, 1, 2]:
                                coeffs_list = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                if sum(1 for c in coeffs_list if c != 0) != 12:
                                    continue
                                coeffs = [(shifted_exps[i], coeffs_list[i]) for i in range(12)]
                                try_candidate(coeffs)
    
    if best_poly is not None:
        return best_poly
    
    return [1]*12

# EVOLVE-BLOCK-END
