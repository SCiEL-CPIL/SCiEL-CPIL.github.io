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
    
    # ====================================================================
    # STRATEGY: Factor-based construction P = A * B * C
    # 
    # Instead of searching 12 coefficients directly, we decompose P into
    # three sparse factors A, B, C with 2-3 terms each. The square P^2 =
    # (A^2)(B^2)(C^2) has structure that can be exploited: if A^2, B^2,
    # C^2 have small support, their convolution may have many cancellations.
    #
    # We enumerate factor structures and use local search on factor coefficients.
    # ====================================================================
    
    def generate_factor_combinations():
        """
        Generate all ways to split 12 terms into 3 factors of size 2-3 each.
        (2,4,6) is impossible since max factor size is 3.
        Valid splits: (2,3,4)? No, max is 3. So (2,3,7)? No.
        Actually with 3 factors each having at most 3 terms: max total = 9.
        We need 12 terms in the product. So factors must have sizes that multiply to >= 12.
        If A has a terms, B has b terms, C has c terms, then P has at most a*b*c terms.
        We need a*b*c >= 12 and the actual product to have exactly 12 nonzero terms.
        
        Possible (a,b,c) with a<=b<=c, a*b*c >= 12:
        (2,2,3): 12 terms max
        (2,2,4): 16 terms max (but c=4 > 3, so not allowed if max factor size is 3)
        
        Wait, the directive says "each factor has 2 or 3 terms". So max product size is 3*3*3=9.
        That's not enough to get 12 terms! 
        
        Let me re-read: "Model the 12-term polynomial as a product of three sparse factors, 
        P(x) = A(x)B(x)C(x), where each factor has 2 or 3 terms"
        
        2*3*3 = 18, 3*3*3 = 27. So we can have up to 27 terms, but we need exactly 12.
        This means some coefficients in the product must cancel to zero.
        
        Valid factor size combinations (a,b,c) with 2<=a<=b<=c<=3 and a*b*c >= 12:
        (2,3,3): 18 terms possible
        (3,3,3): 27 terms possible
        (2,2,3): 12 terms exactly - but then no cancellation needed, just need all terms nonzero
        (2,2,2): 8 terms - too few
        
        So we focus on (2,3,3) and (3,3,3) factor structures.
        """
        pass
    
    # Focus on (2,3,3) factor structure: A has 2 terms, B has 3 terms, C has 3 terms
    # Product has at most 18 terms. We need exactly 12 nonzero terms in P.
    # This means 6 terms must cancel to zero in the product A*B*C.
    
    def try_factor_233():
        """Search for P = A*B*C where A has 2 terms, B has 3, C has 3."""
        # A = a0*x^p0 + a1*x^p1
        # B = b0*x^q0 + b1*x^q1 + b2*x^q2
        # C = c0*x^r0 + c1*x^r1 + c2*x^r2
        
        # For simplicity, start with small exponents.
        # Let's fix the exponent structure and search coefficients.
        
        # Try a specific exponent pattern:
        # A: exponents {0, 1}
        # B: exponents {0, 2, 4}
        # C: exponents {0, 3, 6}
        # Product exponents: all sums p+q+r
        # This gives a structured set of exponents.
        
        exp_A = [0, 1]
        exp_B = [0, 2, 4]
        exp_C = [0, 3, 6]
        
        # The product P has terms at exponents {p+q+r for p in exp_A, q in exp_B, r in exp_C}
        # Max exponent: 1+4+6 = 11
        # Min exponent: 0+0+0 = 0
        
        # Coefficients: a0, a1, b0, b1, b2, c0, c1, c2
        # We want to choose these (integers in [-64,64], nonzero) such that:
        # 1. P = A*B*C has exactly 12 nonzero terms
        # 2. P^2 has minimal nonzero terms
        
        # Let's do local search on the 8 coefficients.
        
        def compute_poly(a0, a1, b0, b1, b2, c0, c1, c2):
            """Compute the polynomial P = A*B*C as a dict {exp: coeff}."""
            A = {exp_A[0]: a0, exp_A[1]: a1}
            B = {exp_B[0]: b0, exp_B[1]: b1, exp_B[2]: b2}
            C = {exp_C[0]: c0, exp_C[1]: c1, exp_C[2]: c2}
            
            # Multiply A * B
            AB = {}
            for ep, cp in A.items():
                for eq, cq in B.items():
                    key = ep + eq
                    AB[key] = AB.get(key, 0) + cp * cq
            
            # Multiply AB * C
            ABC = {}
            for ep, cp in AB.items():
                for eq, cq in C.items():
                    key = ep + eq
                    ABC[key] = ABC.get(key, 0) + cp * cq
            
            return ABC
        
        def get_poly_score(ABC):
            """Count nonzero terms in P^2 where P is given as dict."""
            # Remove zero coefficients
            P = {e: c for e, c in ABC.items() if c != 0}
            if len(P) != 12:
                return float('inf')  # Invalid
            return get_score(P)
        
        # Local search on the 8 coefficients
        def local_search_233(max_iters=50000):
            # Initialize
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(8)]
            curr_poly = compute_poly(*coeffs)
            curr_score = get_poly_score(curr_poly)
            
            if curr_score < best_score:
                update_best(curr_poly, curr_score)
            
            T = 5.0
            for it in range(max_iters):
                if time.time() > deadline:
                    return
                
                # Pick one or two coefficients to modify
                n_mods = random.choice([1, 2])
                indices = random.sample(range(8), n_mods)
                
                old_coeffs = list(coeffs)
                for idx in indices:
                    coeffs[idx] = random.choice([-3, -2, -1, 1, 2, 3])
                
                new_poly = compute_poly(*coeffs)
                new_score = get_poly_score(new_poly)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(new_poly, curr_score)
                else:
                    delta = new_score - curr_score
                    if random.random() < pow(2.718, -delta / max(T, 0.1)):
                        curr_score = new_score
                    else:
                        coeffs = old_coeffs
                
                T *= 0.9995
            
            return curr_score
        
        # Run several restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            random.seed(1000 + restart)
            local_search_233()
    
    # Try (3,3,3) factor structure
    def try_factor_333():
        """Search for P = A*B*C where each factor has 3 terms."""
        # A: exponents {0, 1, 3}
        # B: exponents {0, 2, 5}
        # C: exponents {0, 4, 7}
        # This gives a spread of exponents.
        
        exp_A = [0, 1, 3]
        exp_B = [0, 2, 5]
        exp_C = [0, 4, 7]
        
        def compute_poly(a0, a1, a2, b0, b1, b2, c0, c1, c2):
            A = {exp_A[0]: a0, exp_A[1]: a1, exp_A[2]: a2}
            B = {exp_B[0]: b0, exp_B[1]: b1, exp_B[2]: b2}
            C = {exp_C[0]: c0, exp_C[1]: c1, exp_C[2]: c2}
            
            AB = {}
            for ep, cp in A.items():
                for eq, cq in B.items():
                    key = ep + eq
                    AB[key] = AB.get(key, 0) + cp * cq
            
            ABC = {}
            for ep, cp in AB.items():
                for eq, cq in C.items():
                    key = ep + eq
                    ABC[key] = ABC.get(key, 0) + cp * cq
            
            return ABC
        
        def get_poly_score(ABC):
            P = {e: c for e, c in ABC.items() if c != 0}
            if len(P) != 12:
                return float('inf')
            return get_score(P)
        
        def local_search_333(max_iters=30000):
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(9)]
            curr_poly = compute_poly(*coeffs)
            curr_score = get_poly_score(curr_poly)
            
            if curr_score < best_score:
                update_best(curr_poly, curr_score)
            
            T = 5.0
            for it in range(max_iters):
                if time.time() > deadline:
                    return
                
                n_mods = random.choice([1, 2])
                indices = random.sample(range(9), n_mods)
                
                old_coeffs = list(coeffs)
                for idx in indices:
                    coeffs[idx] = random.choice([-3, -2, -1, 1, 2, 3])
                
                new_poly = compute_poly(*coeffs)
                new_score = get_poly_score(new_poly)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(new_poly, curr_score)
                else:
                    delta = new_score - curr_score
                    if random.random() < pow(2.718, -delta / max(T, 0.1)):
                        curr_score = new_score
                    else:
                        coeffs = old_coeffs
                
                T *= 0.9995
            
            return curr_score
        
        for restart in range(5):
            if time.time() > deadline:
                break
            random.seed(2000 + restart)
            local_search_333()
    
    # Also try the direct approach with a different initialization strategy
    def direct_search_with_structured_init():
        """Direct search on 12 coefficients but with structured initializations."""
        
        # Try initialization based on small integer solutions to quadratic constraints
        # S_2 = 2*c0*c2 + c1^2 = 0 => c2 = -c1^2/(2*c0)
        # For c0=1, c1=2: c2 = -2
        # For c0=1, c1=4: c2 = -8
        # For c0=2, c1=2: c2 = -1
        
        init_patterns = [
            # (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
            [1, 2, -2, 4, -10, 28, -84, 0, 0, 0, 0, 0],  # Will be clipped
            [1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 2, -2, 4, -10, 28, 1, 1, 1, 1, 1, 1],
            [2, 2, -1, 2, -4, 8, 1, 1, 1, 1, 1, 1],
        ]
        
        for init in init_patterns:
            if time.time() > deadline:
                break
            
            # Clip to valid range and ensure nonzero
            curr = {}
            for i, c in enumerate(init):
                c = max(-64, min(64, c))
                if c == 0:
                    c = 1
                curr[i] = c
            
            curr_score = get_score(curr)
            if curr_score < best_score:
                update_best(curr, curr_score)
            
            # Local search from this initialization
            T = 5.0
            for it in range(30000):
                if time.time() > deadline:
                    break
                
                i1 = random.randint(0, 11)
                i2 = random.randint(0, 11)
                if i1 == i2:
                    i2 = (i2 + 1) % 12
                
                old_v1 = curr[i1]
                old_v2 = curr[i2]
                
                new_v1 = random.choice([-3, -2, -1, 1, 2, 3])
                new_v2 = random.choice([-3, -2, -1, 1, 2, 3])
                
                if new_v1 == old_v1 and new_v2 == old_v2:
                    continue
                
                curr[i1] = new_v1
                curr[i2] = new_v2
                
                new_score = get_score(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(curr, curr_score)
                else:
                    delta = new_score - curr_score
                    if random.random() < pow(2.718, -delta / max(T, 0.1)):
                        curr_score = new_score
                    else:
                        curr[i1] = old_v1
                        curr[i2] = old_v2
                
                T *= 0.999
    
    # Run the strategies
    if time.time() < deadline:
        try_factor_233()
    
    if time.time() < deadline:
        try_factor_333()
    
    if time.time() < deadline:
        direct_search_with_structured_init()
    
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
