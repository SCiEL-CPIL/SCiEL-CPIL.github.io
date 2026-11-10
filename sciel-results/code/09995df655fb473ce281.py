<<<<<<< SEARCH
    # Strategy: P = A(x) + x^k * A(x) where A has 6 terms
    if time.time() < deadline:
        for k in range(1, 12):
            for _ in range(80):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    mag = random.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if random.random() < 0.5 else -1
                    A.append(sign * mag)
                
                if k <= 5:
                    full_coeffs = [0] * (max(6, k + 6))
                    for i in range(6):
                        full_coeffs[i] += A[i]
                        full_coeffs[k + i] += A[i]
                    terms = [(i, c) for i, c in enumerate(full_coeffs) if c != 0]
                    if len(terms) != 12:
                        continue
                    coeffs = [c for _, c in terms]
                    exponents = [e for e, _ in terms]
                else:
                    coeffs = A + A
                    exponents = list(range(6)) + list(range(k, k + 6))
                
                if any(abs(c) > 64 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                dm = build_degree_map(exponents)
                nz = fast_square_nonzero(coeffs, exponents, dm)
                update_best(coeffs, exponents, nz)
    
=======
    # Strategy: Factorization P = A * B
    # Try A with 3 terms, B with 4 terms
    if time.time() < deadline:
        # Exponents for A (3 terms)
        A_exp_patterns = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [0, 1, 4],
            [0, 2, 5],
            [0, 3, 6],
            [1, 2, 3],
            [0, 1, 5],
            [0, 2, 6],
            [0, 4, 8],
        ]
        # Exponents for B (4 terms)
        B_exp_patterns = [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 3, 4],
            [0, 2, 4, 6],
            [0, 1, 4, 7],
            [0, 2, 5, 8],
            [0, 3, 6, 9],
            [1, 2, 3, 4],
            [0, 1, 2, 5],
            [0, 2, 4, 8],
        ]
        
        # Coefficient values for factors (small integers)
        coeff_vals = [-3, -2, -1, 1, 2, 3]
        
        for A_exp in A_exp_patterns:
            if time.time() > deadline:
                break
            for B_exp in B_exp_patterns:
                if time.time() > deadline:
                    break
                
                # Iterate over coefficients for A
                for cA in product(coeff_vals, repeat=3):
                    if time.time() > deadline:
                        break
                    # Iterate over coefficients for B
                    for cB in product(coeff_vals, repeat=4):
                        # Construct P = A * B
                        p_coeffs = defaultdict(int)
                        for i in range(3):
                            for j in range(4):
                                deg = A_exp[i] + B_exp[j]
                                p_coeffs[deg] += cA[i] * cB[j]
                        
                        # Check if P has exactly 12 non-zero terms
                        nz_terms = []
                        valid = True
                        for deg, val in p_coeffs.items():
                            if val != 0:
                                if val < -64 or val > 64:
                                    valid = False
                                    break
                                nz_terms.append((deg, val))
                        
                        if not valid or len(nz_terms) != 12:
                            continue
                        
                        # Sort by degree
                        nz_terms.sort()
                        exponents = [t[0] for t in nz_terms]
                        coeffs = [t[1] for t in nz_terms]
                        
                        # Check max degree constraint
                        if max(exponents) * 2 + 1 > 41:
                            continue
                        
                        # Calculate square sparsity
                        dm = build_degree_map(exponents)
                        nz = fast_square_nonzero(coeffs, exponents, dm)
                        update_best(coeffs, exponents, nz)
    
>>>>>>> REPLACE

I also need to import `product` from `itertools`.

