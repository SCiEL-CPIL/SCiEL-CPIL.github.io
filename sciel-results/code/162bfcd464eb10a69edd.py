import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square(coeffs_list):
        """
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: (count_nonzero_in_square, square_list, max_exp)
        """
        if not coeffs_list:
            return 0, [0], 0
        max_e = max(e for e, c in coeffs_list)
        square = [0] * (2 * max_e + 1)
        cl = sorted(coeffs_list)
        for i in range(12):
            ei, ci = cl[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, 12):
                ej, cj = cl[j]
                square[ei + ej] += 2 * ci * cj
        count = sum(1 for x in square if x != 0)
        return count, square, max_e

    def to_full(coeffs_list):
        if not coeffs_list:
            return None
        max_e = max(e for e, c in coeffs_list)
        if max_e >= 41:
            return None
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        nz = sum(1 for x in full if x != 0)
        if nz != 12:
            return None
        if any(abs(x) > 64 for x in full):
            return None
        return full

    def try_update(count, coeffs_list):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = to_full(coeffs_list)
            if full is not None:
                best_count = count
                best_coeffs = full
                return True
        return False

    def smooth_score(coeffs_list):
        """Smooth surrogate: count + small penalty for large coefficients."""
        count, _, _ = eval_square(coeffs_list)
        penalty = sum(1.0 / max(1, abs(c)) for _, c in coeffs_list) * 0.01
        return count + penalty

    def hill_climb(start_cl, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        if len(cl) != 12:
            return cl, float('inf')
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        no_improve_count = 0
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16]
        while time.time() - t0 < time_budget:
            improved = False
            for _ in range(5):
                idx = random.randrange(12)
                e, c = cl[idx]
                if random.random() < 0.7:
                    new_c = random.choice(vals)
                    if new_c != c and abs(new_c) <= 64:
                        cl2 = list(cl)
                        cl2[idx] = (e, new_c)
                        c2, _, _ = eval_square(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps = [cl[i][0] for i in range(12)]
                        exps.remove(e)
                        if new_e not in exps:
                            cl2 = list(cl)
                            cl2[idx] = (new_e, c)
                            c2, _, _ = eval_square(cl2)
                            if c2 < count:
                                cl = cl2
                                count = c2
                                try_update(count, cl)
                                improved = True
                                no_improve_count = 0
                                break
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    for _ in range(2):
                        idx = random.randrange(12)
                        e, c = cl[idx]
                        new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                        if new_c != c and abs(new_c) <= 64:
                            cl[idx] = (e, new_c)
                    no_improve_count = 0
                    count, _, _ = eval_square(cl)
                    try_update(count, cl)
            if best_count == 12:
                break
        return cl, count

    def convolve_sparse(poly1, poly2):
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]

    # =====================================================================
    # Phase 1: Constraint Satisfaction over Square Support
    # Fix a target sparse support S for Q = P^2, solve for P's coefficients
    # =====================================================================
    
    def solve_for_p_given_square_support(target_support, max_degree=40):
        """
        Given a target support S (list of exponents where Q is nonzero),
        try to find P with 12 nonzero terms such that P^2 has support subset of S.
        Uses sign consistency and integer factorization.
        """
        if time.time() > deadline - 15:
            return None
        
        # For each pair of exponents in P, the sum must be in S
        # We search over P's exponent set (12 exponents) and coefficient signs/values
        
        # Start with small exponent sets
        base_exps = list(range(0, min(15, max_degree + 1)))
        
        # Try all 12-subsets of base_exps (combinatorial, but pruned)
        # For efficiency, try structured subsets first
        
        # Strategy 1: Consecutive exponents
        for start in range(0, max(1, 41 - 12)):
            exps = list(range(start, start + 12))
            if max(exps) > max_degree:
                break
            
            # Try sign patterns (2^12 = 4096)
            for signs in product([-1, 1], repeat=12):
                if time.time() > deadline - 15:
                    return None
                cl = [(exps[i], signs[i]) for i in range(12)]
                count, sq, _ = eval_square(cl)
                # Check if support is subset of target
                sq_set = set(i for i, v in enumerate(sq) if v != 0)
                if sq_set.issubset(set(target_support)):
                    try_update(count, cl)
                    if best_count <= len(target_support):
                        return cl
        return None

    # Generate candidate target supports
    # Geometric progressions, arithmetic progressions, etc.
    target_supports = []
    
    # Arithmetic progressions
    for start in range(0, 5):
        for step in range(1, 6):
            support = list(range(start, min(41, start + 20 * step), step))
            if len(support) >= 12:
                target_supports.append(support[:12])
    
    # Geometric progressions
    for base in range(1, 4):
        support = [base ** i for i in range(12) if base ** i <= 40]
        if len(support) >= 12:
            target_supports.append(support[:12])
    
    # Try solving for each target support
    for ts in target_supports[:20]:  # Limit to avoid timeout
        if time.time() > deadline - 20:
            break
        result = solve_for_p_given_square_support(ts)
        if result is not None:
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: Structural Search via Factorization
    # =====================================================================
    
    candidate_factors_pool = []
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                if a == 0 or b == 0: continue
                candidate_factors_pool.append([(0, a), (k, b)])
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a==0 or b==0 or c==0: continue
                    candidate_factors_pool.append([(0, a), (k, b), (2*k, c)])
                    if k + 1 <= 15:
                        candidate_factors_pool.append([(0, a), (k, b), (k+1, c)])
    
    pool_2 = [f for f in candidate_factors_pool if len(f) == 2]
    pool_3 = [f for f in candidate_factors_pool if len(f) == 3]
    
    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)):
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4:
                pair_products.append(prod)
    
    seen_polys = set()
    for pp in pair_products:
        if time.time() > deadline - 25:
            break
        for c in pool_3:
            max_e_pp = max(e for e, _ in pp)
            max_e_c = max(e for e, _ in c)
            if max_e_pp + max_e_c > 40:
                continue
            p_poly = convolve_sparse(pp, c)
            if len(p_poly) != 12:
                continue
            p_sorted = sorted(p_poly)
            key = tuple(p_sorted)
            if key in seen_polys:
                continue
            seen_polys.add(key)
            count, _, _ = eval_square(p_sorted)
            try_update(count, p_sorted)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 3: Local Search
    # =====================================================================
    random.seed(42)
    
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        hill_climb(start_cl, time_budget=5.0)
        if best_count == 12:
            return best_coeffs
    
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        if random.random() < 0.5:
            exps = list(range(12))
        else:
            exps = sorted(random.sample(range(40), 12))
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        for i in range(12):
            if cl[i][1] == 0:
                cl[i] = (cl[i][0], 1)
        hill_climb(cl, time_budget=1.0)
        if best_count == 12:
            return best_coeffs
        if attempts % 10 == 0:
            base_exp = random.randint(1, 3)
            cl = [(i * base_exp, random.choice(vals)) for i in range(12)]
            if cl[-1][0] > 40:
                continue
            hill_climb(cl, time_budget=0.5)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 4: Final Polish with Tighter Moves
    # =====================================================================
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        def hill_climb_polish(start_cl, time_budget=5.0):
            nonlocal best_count, best_coeffs
            t0 = time.time()
            cl = [tuple(x) for x in start_cl]
            count, _, _ = eval_square(cl)
            try_update(count, cl)
            vals = [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]
            while time.time() - t0 < time_budget:
                improved = False
                for _ in range(10):
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    new_c = random.choice(vals)
                    if new_c != c:
                        cl2 = list(cl)
                        cl2[idx] = (e, new_c)
                        c2, _, _ = eval_square(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
                            break
                if not improved:
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    shift = random.choice([-1, 1])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps = [cl[i][0] for i in range(12)]
                        exps.remove(e)
                        if new_e not in exps:
                            cl2 = list(cl)
                            cl2[idx] = (new_e, c)
                            c2, _, _ = eval_square(cl2)
                            if c2 < count:
                                cl = cl2
                                count = c2
                                try_update(count, cl)
                                improved = True
                if best_count == 12:
                    break
            return cl, count
        hill_climb_polish(start_cl, time_budget=5.0)
        if best_count == 12:
            return best_coeffs

    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29
        if len(best_coeffs) > 41:
            best_coeffs = best_coeffs[:41]
        if best_coeffs[-1] == 0:
            for i in range(len(best_coeffs)-1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i+1]
                    break
            if not best_coeffs:
                best_coeffs = [1]
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            best_coeffs = [0]*41
            for i in range(12):
                best_coeffs[i] = 1

    return best_coeffs

    # =====================================================================
    # Phase 1: Structural Search via Factorization
    # Strategy: P = A * B * C. 
    # We search for A, B, C with small support such that P has exactly 12 terms.
    # Common structures:
    # 1. A = a0 + a1 x^k (2 terms)
    # 2. B = b0 + b1 x^m (2 terms)
    # 3. C = c0 + c1 x^n + c2 x^p (3 terms)
    # Or 2x2x3, 2x3x2, etc.
    # The square P^2 = A^2 B^2 C^2.
    # If A, B, C are chosen with specific signs, cancellations in A^2, B^2, C^2 
    # or in the convolution can reduce support.
    # =====================================================================
    
    # Define a set of candidate sparse polynomials for factors
    # We will generate P directly from these factors and check if it has 12 terms.
    
    def convolve_sparse(poly1, poly2):
        """
        poly1, poly2: lists of (exp, coef)
        Returns list of (exp, coef) for product, removing zeros.
        """
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]
    
    def get_poly_from_factors(factors):
        """
        factors: list of sparse polys (list of (exp, coef))
        Returns sparse poly for product.
        """
        if not factors:
            return [(0, 1)]
        res = factors[0]
        for f in factors[1:]:
            res = convolve_sparse(res, f)
            if len(res) > 12: # Early exit if too many terms
                return None
        return res

    # Generate candidate factor sets
    # We focus on small exponents to keep degree <= 40.
    # Max degree of P is sum of max degrees of factors.
    # If max deg is 40, and we have 3 factors, avg max deg ~ 13.
    
    candidate_factors_pool = []
    
    # Type 1: 2-term factors a + b x^k
    # Coeffs in {-1, 1, 2, -2}
    # Exponents 1..10
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                if a == 0 or b == 0: continue
                # Normalize: make first non-zero coeff positive? 
                # Actually, global sign doesn't matter for square count, but relative sign does.
                # We keep all to be safe, but might generate duplicates.
                candidate_factors_pool.append([(0, a), (k, b)])
                
    # Type 2: 3-term factors a + b x^k + c x^m
    # To keep it manageable, we restrict structure:
    # a + b x^k + c x^(2k) (Geometric)
    # a + b x^k + c x^m (Arbitrary small m)
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a==0 or b==0 or c==0: continue
                    # Geometric
                    candidate_factors_pool.append([(0, a), (k, b), (2*k, c)])
                    # Linear-ish
                    if k + 1 <= 15:
                        candidate_factors_pool.append([(0, a), (k, b), (k+1, c)])
    
    # We need to combine 3 factors to get 12 terms.
    # Possibilities for term counts (n1, n2, n3):
    # 2 * 2 * 3 = 12
    # 2 * 3 * 2 = 12
    # 3 * 2 * 2 = 12
    # 2 * 2 * 2 = 8 (Too few, need to add terms? No, P must have exactly 12 terms)
    # 2 * 2 * 4 = 16 (Too many)
    # 2 * 3 * 3 = 18 (Too many)
    # So we MUST use 2, 2, 3 term factors.
    
    # To avoid O(N^3) explosion, we pre-compute pairs of 2-term factors?
    # Or just iterate smartly.
    # Pool size is roughly 40 (2-term) + 200 (3-term) = 240.
    # 240 * 240 * 240 is ~13M. Too slow in Python if we do full convolve each time.
    # We need to prune.
    
    # Strategy: 
    # 1. Iterate over pairs of 2-term factors (A, B). Compute AB.
    #    AB has at most 4 terms.
    # 2. Iterate over 3-term factors (C).
    # 3. Compute (AB) * C. Check if it has exactly 12 terms.
    
    # Let's separate pools
    pool_2 = [f for f in candidate_factors_pool if len(f) == 2]
    pool_3 = [f for f in candidate_factors_pool if len(f) == 3]
    
    # Pre-compute products of pairs from pool_2
    # There are ~40 in pool_2. 40*40 = 1600 pairs. This is fast.
    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)): # Symmetry
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4: # Should be <=4
                pair_products.append(prod)
                
    # Now iterate over pair_products and pool_3
    # 1600 * 200 = 320,000 iterations.
    # Each iteration does a convolve of size 4 and 3 -> max 12 terms.
    # This is feasible within a few seconds.
    
    seen_polys = set()
    
    for pp in pair_products:
        if time.time() > deadline - 10:
            break
        for c in pool_3:
            # Check degree bound early
            max_e_pp = max(e for e, _ in pp)
            max_e_c = max(e for e, _ in c)
            if max_e_pp + max_e_c > 40:
                continue
            
            p_poly = convolve_sparse(pp, c)
            if len(p_poly) != 12:
                continue
            
            # Normalize to avoid duplicates
            # Sort by exponent
            p_sorted = sorted(p_poly)
            key = tuple(p_sorted)
            
            if key in seen_polys:
                continue
            seen_polys.add(key)
            
            # Evaluate square
            count, _, _ = eval_square(p_sorted)
            try_update(count, p_sorted)
            
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: Local Search from Structural Seeds and Random Starts
    # =====================================================================
    
    random.seed(42)
    
    # Start from best found so far
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        hill_climb(start_cl, time_budget=5.0)
        if best_count == 12:
            return best_coeffs
    
    # Start from random 12-term polynomials
    # Focus on consecutive exponents first, as they are dense and allow good cancellation
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    # Run multiple random starts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Random exponents, mostly consecutive or small gaps
        if random.random() < 0.5:
            exps = list(range(12))
        else:
            exps = sorted(random.sample(range(40), 12))
        
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        # Ensure non-zero
        for i in range(12):
            if cl[i][1] == 0:
                cl[i] = (cl[i][0], 1)
                
        hill_climb(cl, time_budget=1.0)
        if best_count == 12:
            return best_coeffs
        
        # Occasionally try a "geometric" random start
        if attempts % 10 == 0:
            base_exp = random.randint(1, 3)
            cl = [(i * base_exp, random.choice(vals)) for i in range(12)]
            # Check degree
            if cl[-1][0] > 40:
                continue
            hill_climb(cl, time_budget=0.5)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 3: Final Polish
    # If we found a good candidate, try to refine it further with tighter moves
    # =====================================================================
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        # Use a more aggressive hill climb with larger coefficient jumps
        def hill_climb_polish(start_cl, time_budget=5.0):
            nonlocal best_count, best_coeffs
            t0 = time.time()
            cl = [tuple(x) for x in start_cl]
            count, _, _ = eval_square(cl)
            try_update(count, cl)
            
            vals = [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]
            
            while time.time() - t0 < time_budget:
                improved = False
                for _ in range(10):
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    new_c = random.choice(vals)
                    if new_c != c:
                        cl2 = list(cl)
                        cl2[idx] = (e, new_c)
                        c2, _, _ = eval_square(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
                            break
                if not improved:
                    # Try exponent shift
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    shift = random.choice([-1, 1])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps = [cl[i][0] for i in range(12)]
                        exps.remove(e)
                        if new_e not in exps:
                            cl2 = list(cl)
                            cl2[idx] = (new_e, c)
                            c2, _, _ = eval_square(cl2)
                            if c2 < count:
                                cl = cl2
                                count = c2
                                try_update(count, cl)
                                improved = True
                if best_count == 12:
                    break
            return cl, count

        hill_climb_polish(start_cl, time_budget=5.0)
        if best_count == 12:
            return best_coeffs

    # Fallback
    if best_coeffs is None:
        # Construct a valid 12-term polynomial
        # e.g., x^0 - x^1 + x^2 ...
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29
        # Verify
        if len(best_coeffs) > 41:
            best_coeffs = best_coeffs[:41]
        if best_coeffs[-1] == 0:
            # Ensure last is non-zero
            # Find last non-zero
            for i in range(len(best_coeffs)-1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i+1]
                    break
            if not best_coeffs:
                best_coeffs = [1]
        
        # Ensure exactly 12 non-zeros
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            # Just create a simple one
            best_coeffs = [0]*41
            for i in range(12):
                best_coeffs[i] = 1
            # This has 12 non-zeros. Square will be dense, but it's valid.
            
    return best_coeffs
