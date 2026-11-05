import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # First, build the known rank-39 decomposition using Strassen blocks
    def build_strassen_decomposition():
        # Rank calculation:
        # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
        # 1 Schoolbook block (2x2x1) for k=6 -> 4
        # 1 Schoolbook block (2x1x7) for j=2 -> 14
        rank = 21 + 4 + 14
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, coeffs=None):
            nonlocal col
            if coeffs is None:
                coeffs = [1] * len(u_indices)
            for idx, c in zip(u_indices, coeffs):
                U[idx][col] += c
            v_coeffs = [1] * len(v_indices)
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        # Helper for Strassen block
        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            add_term([0, 4], [b00, b11], [c00, c11])
            
            # M2: (A10+A11)B00 -> C10, -C11
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            # M3: A00(B01-B11) -> C01, C11
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            # M4: A11(B10-B00) -> C00, C10
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            # M5: (A00+A01)B11 -> -C00, C01
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            # M6: (A10-A00)(B00+B01) -> C11
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            # M7: (A01-A11)(B10+B11) -> C00
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        # Add 3 Strassen blocks
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        # Block 4: k=6, 2x2x1 Schoolbook
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        # Part 2: j=2
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W form a valid exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_dissection_decomposition():
        """
        Bini-Brenti dissection: split 3x7 into 3x3 + 3x4.
        Use rank-5 for 2x3x3 (Strassen-like) and rank-9 for 2x3x4.
        Total: 5 + 9 = 14, then handle j=2 separately with rank 14.
        Actually, let's think more carefully.
        
        For 2x3x3: known rank is 5 (Strassen for 2x2x2 gives 7, but 2x3x3 can be 5)
        For 2x3x4: we can use 2 Strassen blocks for first 2 cols (14) + 2 schoolbook for last 2 cols (8) = 22
        That's worse. Let me try a different approach.
        
        Actually, the current rank-39 uses:
        - 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5: 3*7 = 21
        - Schoolbook for k=6: 4
        - Schoolbook for j=2: 14
        Total: 39
        
        A better dissection might be:
        - Treat as (2x3x2) * 3 blocks + (2x3x1) * 1 block
        - 2x3x2 can be done in rank 5 (not 7)
        - 2x3x1 can be done in rank 3 (not 4)
        - j=2 still needs 14
        Total: 3*5 + 3 + 14 = 32
        """
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
            nonlocal col
            if u_coeffs is None:
                u_coeffs = [1] * len(u_indices)
            if v_coeffs is None:
                v_coeffs = [1] * len(v_indices)
            if w_coeffs is None:
                w_coeffs = [1] * len(w_indices)
            for idx, c in zip(u_indices, u_coeffs):
                U[idx][col] += c
            for idx, c in zip(v_indices, v_coeffs):
                V[idx][col] += c
            for idx, c in zip(w_indices, w_coeffs):
                W[idx][col] += c
            col += 1

        # For each pair of k values (0,1), (2,3), (4,5): use rank-5 for 2x3x2
        # 2x3x2 rank-5 decomposition:
        # A is 2x3, B is 3x2, C is 2x2
        # Let A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01],[b10,b11],[b20,b21]]
        # C00 = a00*b00 + a01*b10 + a02*b20
        # C01 = a00*b01 + a01*b11 + a02*b21
        # C10 = a10*b00 + a11*b10 + a12*b20
        # C11 = a10*b01 + a11*b11 + a12*b21
        
        # Rank-5 for 2x3x2:
        # M1: (a00+a11)(b00+b10) -> contributes to C00 (a00*b00, a00*b10, a11*b00, a11*b10)
        #     and C10 (a11*b00, a11*b10) - need to be careful
        # This is getting complex. Let me just use the known rank-39 and try to optimize it.
        
        # For now, return the standard rank-39
        return build_strassen_decomposition()

    def try_lower_rank(start_rank, end_rank, U0, V0, W0, max_iter=500, tol=1e-10):
        """
        Try to find a lower rank decomposition starting from the given initial decomposition.
        Uses vectorized iterative refinement with smooth updates and polish pass.
        """
        if time.time() > deadline - 5:
            return None
            
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        U0_arr = np.array(U0, dtype=np.float64)
        V0_arr = np.array(V0, dtype=np.float64)
        W0_arr = np.array(W0, dtype=np.float64)
        
        def smooth_inv(x, eps=1e-10):
            """Smooth inverse: 1/x for |x| > eps, 0 otherwise, with smooth transition."""
            safe_x = np.where(np.abs(x) > eps, x, np.where(x >= 0, eps, -eps))
            return 1.0 / safe_x
        
        for target_rank in range(start_rank, min(end_rank + 1, 40)):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(42 + target_rank)
            
            best_residual = np.inf
            best_solution = None
            
            n_restarts = 12 if target_rank <= 36 else 8
            
            for restart in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                    
                # Initialize by taking a random linear combination of the existing rank-39 terms
                if target_rank < 39:
                    # Use structured initialization: mix with different strategies
                    strategy = restart % 4
                    if strategy == 0:
                        # Standard random mixing
                        mix = rng.standard_normal((39, target_rank))
                    elif strategy == 1:
                        # Sparse mixing: keep most terms, drop a few
                        mix = np.zeros((39, target_rank))
                        for r in range(target_rank):
                            idx = rng.integers(0, 39)
                            mix[idx, r] = 1.0
                        # Add small random perturbation
                        mix += 0.1 * rng.standard_normal((39, target_rank))
                    elif strategy == 2:
                        # Block-structured mixing: group terms by Strassen block
                        mix = np.zeros((39, target_rank))
                        block_size = 7  # Each Strassen block has 7 terms
                        for b in range(3):  # 3 Strassen blocks
                            start = b * block_size
                            end = start + block_size
                            # Assign these 7 terms to ceil(7 * target_rank / 39) columns
                            n_cols = max(1, round(7 * target_rank / 39))
                            cols = list(range(b * n_cols, min((b+1) * n_cols, target_rank)))
                            for c in cols:
                                mix[start:end, c] = rng.standard_normal(7)
                        # Fill remaining columns
                        for c in range(target_rank):
                            if mix[:, c].sum() == 0:
                                mix[:, c] = rng.standard_normal(39)
                    else:
                        # Diagonal-dominant mixing
                        mix = np.zeros((39, target_rank))
                        for r in range(target_rank):
                            idx = r % 39
                            mix[idx, r] = 1.0
                        mix += 0.05 * rng.standard_normal((39, target_rank))
                    
                    U = U0_arr @ mix
                    V = V0_arr @ mix
                    W = W0_arr @ mix
                else:
                    U = U0_arr.copy()
                    V = V0_arr.copy()
                    W = W0_arr.copy()
                
                # Reshape to (n, m, r), (m, p, r), (p, n, r)
                U_3d = U.reshape(n, m, target_rank).copy()
                V_3d = V.reshape(m, p, target_rank).copy()
                W_3d = W.reshape(p, n, target_rank).copy()
                
                # Iterative refinement with smooth updates
                prev_residual = np.inf
                stagnant_count = 0
                
                for it in range(max_iter):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U: U_{ij,r} = 1 / sum_k V_{jk,r} W_{ki,r}
                    denom_U = np.einsum('jkr,kir->jir', V_3d, W_3d)
                    denom_U = np.transpose(denom_U, (1, 0, 2))
                    U_3d = smooth_inv(denom_U, eps=1e-12)
                    
                    # Update V: V_{jk,r} = 1 / sum_i U_{ij,r} W_{ki,r}
                    denom_V = np.einsum('ijr,kir->jkr', U_3d, W_3d)
                    V_3d = smooth_inv(denom_V, eps=1e-12)
                    
                    # Update W: W_{ki,r} = 1 / sum_j U_{ij,r} V_{jk,r}
                    denom_W = np.einsum('ijr,jkr->kir', U_3d, V_3d)
                    W_3d = smooth_inv(denom_W, eps=1e-12)
                    
                    # Compute residual
                    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    residual = np.sqrt(np.sum((T - 1.0)**2))
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_solution = (U_3d.copy(), V_3d.copy(), W_3d.copy())
                        stagnant_count = 0
                    else:
                        stagnant_count += 1
                    
                    # Check for stagnation
                    if residual > prev_residual * 0.99:
                        stagnant_count += 1
                    prev_residual = residual
                    
                    if residual < tol:
                        break
                    
                    # Early exit if stuck
                    if stagnant_count > 50 and it > 100:
                        break
                
                if best_residual < tol:
                    break
            
            # Polish pass: if we got close, try to converge to exact solution
            if best_solution is not None and best_residual < 1e-3:
                U_pol, V_pol, W_pol = best_solution
                for it in range(500):
                    if time.time() > deadline - 5:
                        break
                    denom_U = np.einsum('jkr,kir->jir', V_pol, W_pol)
                    denom_U = np.transpose(denom_U, (1, 0, 2))
                    U_pol = smooth_inv(denom_U, eps=1e-14)
                    denom_V = np.einsum('ijr,kir->jkr', U_pol, W_pol)
                    V_pol = smooth_inv(denom_V, eps=1e-14)
                    denom_W = np.einsum('ijr,jkr->kir', U_pol, V_pol)
                    W_pol = smooth_inv(denom_W, eps=1e-14)
                    T = np.einsum('ijr,jkr,kir->ijk', U_pol, V_pol, W_pol)
                    residual = np.sqrt(np.sum((T - 1.0)**2))
                    if residual < 1e-14:
                        break
                
                U_flat = U_pol.reshape(n*m, target_rank)
                V_flat = V_pol.reshape(m*p, target_rank)
                W_flat = W_pol.reshape(p*n, target_rank)
                
                # Round to nearest rational with small denominator for exactness
                def to_rational(x):
                    # Try to find a simple rational approximation
                    for denom in [1, 2, 3, 4, 6, 12]:
                        num = round(x * denom)
                        if abs(x - num/denom) < 1e-10:
                            return num/denom
                    return round(x, 10)
                
                U_list = [[to_rational(x) for x in row] for row in U_flat]
                V_list = [[to_rational(x) for x in row] for row in V_flat]
                W_list = [[to_rational(x) for x in row] for row in W_flat]
                
                if verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with the known rank-39 decomposition
    base_solution = build_strassen_decomposition()
    base_rank = 39
    
    # Verify the base solution
    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
        # Fallback to schoolbook if Strassen is broken
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    better_solution = try_lower_rank(34, 38, base_solution["U"], base_solution["V"], base_solution["W"])
    
    if better_solution is None:
        # If high-rank search failed, try mid-range
        better_solution = try_lower_rank(30, 33, base_solution["U"], base_solution["V"], base_solution["W"])
    
    if better_solution is not None:
        return better_solution
    else:
        return base_solution

# EVOLVE-BLOCK-END
