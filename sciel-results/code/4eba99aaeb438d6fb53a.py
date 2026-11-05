import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = float('inf')
    best_poly = None
    
    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            max_exp = max(poly_dict.keys())
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                lst[e] = c
            best_poly = lst
    
    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)
    
    def is_valid(poly_dict):
        if len(poly_dict) != 12:
            return False
        for c in poly_dict.values():
            if not (-64 <= c <= 64) or c == 0:
                return False
        if max(poly_dict.keys()) > 40:
            return False
        return True
    
    # ====================================================================
    # STRATEGY: Factor-based construction P = A * B * C
    #
    # Instead of searching 12 coefficients directly, we decompose P into
    # three sparse factors A, B, C with 2-3 terms each. The square P^2 =
    # (A^2)(B^2)(C^2) has structure that can be exploited: if A^2, B^2,
    # C^2 have small support, their convolution may have many cancellations.
    #
    # We enumerate factor structures and use local search on coefficients.
    # ====================================================================
    
    def compute_square_from_factors(factors):
        """Compute P = product of factors, then count nonzero terms in P^2."""
        # factors is a list of (exponent, coefficient) tuples for each factor
        # First compute P = A * B * C
        # Start with first factor
        p_dict = {}
        for exp, coeff in factors[0]:
            p_dict[exp] = coeff
        
        # Multiply by subsequent factors
        for f in factors[1:]:
            new_p = {}
            for e1, c1 in p_dict.items():
                for e2, c2 in f:
                    key = e1 + e2
                    new_p[key] = new_p.get(key, 0) + c1 * c2
            p_dict = new_p
        
        # Now p_dict is P
        # Check validity
        if len(p_dict) != 12:
            return None, float('inf')
        for c in p_dict.values():
            if c == 0:
                return None, float('inf')
            if not (-64 <= c <= 64):
                return None, float('inf')
        if max(p_dict.keys()) > 40:
            return None, float('inf')
        
        # Compute square
        score = get_score(p_dict)
        return p_dict, score
    
    def local_search_233(factors, max_iters=2000):
        """Local search on coefficients of 2-3-3 factor structure."""
        # factors: [list_of_2_terms, list_of_3_terms, list_of_3_terms]
        # Each term is (exp, coeff)
        curr_factors = [list(f) for f in factors]
        
        # Compute current P and score
        p_dict, curr_score = compute_square_from_factors(curr_factors)
        if p_dict is None:
            return float('inf'), None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random factor and term to modify
            f_idx = random.choice([0, 1, 2])
            term_idx = random.randrange(len(curr_factors[f_idx]))
            
            old_exp, old_coeff = curr_factors[f_idx][term_idx]
            
            # Try modifying coefficient
            new_coeff = old_coeff + random.choice([-2, -1, 1, 2])
            if -64 <= new_coeff <= 64 and new_coeff != 0:
                curr_factors[f_idx][term_idx] = (old_exp, new_coeff)
            else:
                curr_factors[f_idx][term_idx] = (old_exp, random.choice([-3, -2, -1, 1, 2, 3]))
            
            p_dict, new_score = compute_square_from_factors(curr_factors)
            if p_dict is None:
                curr_factors[f_idx][term_idx] = (old_exp, old_coeff)
                continue
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    update_best(p_dict, curr_score)
            else:
                # Revert
                curr_factors[f_idx][term_idx] = (old_exp, old_coeff)
            
            # Occasionally try modifying exponent
            if random.random() < 0.1:
                f_idx = random.choice([0, 1, 2])
                term_idx = random.randrange(len(curr_factors[f_idx]))
                old_exp, old_coeff = curr_factors[f_idx][term_idx]
                new_exp = old_exp + random.choice([-1, 1, -2, 2])
                if 0 <= new_exp <= 40:
                    curr_factors[f_idx][term_idx] = (new_exp, old_coeff)
                    p_dict, new_score = compute_square_from_factors(curr_factors)
                    if p_dict is not None and new_score < curr_score:
                        curr_score = new_score
                        if curr_score < best_score:
                            update_best(p_dict, curr_score)
                    else:
                        curr_factors[f_idx][term_idx] = (old_exp, old_coeff)
        
        return curr_score, p_dict
    
    def try_factor_233():
        """Search over 2-3-3 factor structures."""
        # Factor A: 2 terms, Factor B: 3 terms, Factor C: 3 terms
        # Exponents for A: (a1, a2), B: (b1, b2, b3), C: (c1, c2, c3)
        # P has exponents that are sums of one from each factor
        # We want P to have exactly 12 terms
        
        # Start with some structured exponent patterns
        base_exps_A = [(0, 0), (0, 1), (1, 2), (0, 2), (1, 3), (2, 4)]
        base_exps_B = [(0, 1, 2), (0, 2, 4), (1, 2, 3), (0, 1, 3), (2, 3, 4), (0, 3, 6)]
        base_exps_C = [(0, 1, 2), (0, 2, 4), (1, 2, 3), (0, 1, 3), (2, 3, 4), (0, 3, 6)]
        
        # Coefficient patterns
        coeff_options = [-3, -2, -1, 1, 2, 3]
        
        # Systematic search over exponent combinations
        for exps_A in base_exps_A:
            if time.time() > deadline:
                return
            for exps_B in base_exps_B:
                if time.time() > deadline:
                    return
                for exps_C in base_exps_C:
                    if time.time() > deadline:
                        return
                    
                    # Build initial factors with random coefficients
                    for trial in range(50):
                        if time.time() > deadline:
                            return
                        
                        factors = [
                            [(exps_A[0], random.choice(coeff_options)),
                             (exps_A[1], random.choice(coeff_options))],
                            [(exps_B[0], random.choice(coeff_options)),
                             (exps_B[1], random.choice(coeff_options)),
                             (exps_B[2], random.choice(coeff_options))],
                            [(exps_C[0], random.choice(coeff_options)),
                             (exps_C[1], random.choice(coeff_options)),
                             (exps_C[2], random.choice(coeff_options))]
                        ]
                        
                        p_dict, score = compute_square_from_factors(factors)
                        if p_dict is not None:
                            if score < best_score:
                                update_best(p_dict, score)
                            if time.time() < deadline:
                                local_search_233(factors, max_iters=500)
    
    def try_factor_224():
        """Search over 2-2-4 factor structures."""
        # Factor A: 2 terms, Factor B: 2 terms, Factor C: 4 terms
        base_exps_A = [(0, 0), (0, 1), (1, 2), (0, 2)]
        base_exps_B = [(0, 1), (1, 2), (0, 2), (2, 3)]
        base_exps_C = [(0, 1, 2, 3), (0, 2, 4, 6), (1, 2, 3, 4), (0, 1, 3, 5)]
        coeff_options = [-3, -2, -1, 1, 2, 3]
        
        for exps_A in base_exps_A:
            if time.time() > deadline:
                return
            for exps_B in base_exps_B:
                if time.time() > deadline:
                    return
                for exps_C in base_exps_C:
                    if time.time() > deadline:
                        return
                    
                    for trial in range(30):
                        if time.time() > deadline:
                            return
                        
                        factors = [
                            [(exps_A[0], random.choice(coeff_options)),
                             (exps_A[1], random.choice(coeff_options))],
                            [(exps_B[0], random.choice(coeff_options)),
                             (exps_B[1], random.choice(coeff_options))],
                            [(exps_C[0], random.choice(coeff_options)),
                             (exps_C[1], random.choice(coeff_options)),
                             (exps_C[2], random.choice(coeff_options)),
                             (exps_C[3], random.choice(coeff_options))]
                        ]
                        
                        p_dict, score = compute_square_from_factors(factors)
                        if p_dict is not None:
                            if score < best_score:
                                update_best(p_dict, score)
                            if time.time() < deadline:
                                local_search_233(factors, max_iters=300)
    
    def try_factor_333():
        """Search over 3-3-3 factor structures."""
        base_exps = [(0, 1, 2), (0, 2, 4), (1, 2, 3), (0, 1, 3), (2, 3, 4), (0, 3, 6)]
        coeff_options = [-3, -2, -1, 1, 2, 3]
        
        for exps_A in base_exps:
            if time.time() > deadline:
                return
            for exps_B in base_exps:
                if time.time() > deadline:
                    return
                for exps_C in base_exps:
                    if time.time() > deadline:
                        return
                    
                    for trial in range(20):
                        if time.time() > deadline:
                            return
                        
                        factors = [
                            [(exps_A[0], random.choice(coeff_options)),
                             (exps_A[1], random.choice(coeff_options)),
                             (exps_A[2], random.choice(coeff_options))],
                            [(exps_B[0], random.choice(coeff_options)),
                             (exps_B[1], random.choice(coeff_options)),
                             (exps_B[2], random.choice(coeff_options))],
                            [(exps_C[0], random.choice(coeff_options)),
                             (exps_C[1], random.choice(coeff_options)),
                             (exps_C[2], random.choice(coeff_options))]
                        ]
                        
                        p_dict, score = compute_square_from_factors(factors)
                        if p_dict is not None:
                            if score < best_score:
                                update_best(p_dict, score)
                            if time.time() < deadline:
                                local_search_233(factors, max_iters=300)
    
    # Phase 1: Factor-based searches
    try_factor_233()
    try_factor_224()
    try_factor_333()
    
    # Phase 2: Direct local search from best found so far
    if best_poly is not None and time.time() < deadline:
        final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
        if len(final_dict) == 12:
            # Hill climb directly on the polynomial
            curr = dict(final_dict)
            curr_score = get_score(curr)
            
            for it in range(20000):
                if time.time() > deadline:
                    break
                
                # Pick 1-2 coefficients to modify
                n_mods = random.choice([1, 1, 2])
                keys = list(curr.keys())
                indices = random.sample(range(len(keys)), n_mods)
                
                old_vals = [curr[keys[i]] for i in indices]
                for idx in indices:
                    k = keys[idx]
                    old = curr[k]
                    new = old + random.choice([-2, -1, 1, 2])
                    if -64 <= new <= 64 and new != 0:
                        curr[k] = new
                    else:
                        curr[k] = random.choice([-3, -2, -1, 1, 2, 3])
                
                new_score = get_score(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(curr, curr_score)
                else:
                    for idx, old in zip(indices, old_vals):
                        curr[keys[idx]] = old
    
    # Fallback
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    if len(final_dict) != 12:
        best_poly = [1] * 12
    else:
        for c in final_dict.values():
            if not (-64 <= c <= 64):
                best_poly = [1] * 12
                break
        max_exp = max(final_dict.keys())
        if max_exp > 40:
            best_poly = [1] * 12
    
    return best_poly
