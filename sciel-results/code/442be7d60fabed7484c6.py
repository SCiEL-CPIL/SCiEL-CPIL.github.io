def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - _t0 < _DEADLINE

    def build_rank66():
        """Build the rank-66 Strassen+schoolbook decomposition."""
        terms = []
        
        def add_schoolbook(i, j, k):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(i, k): 1}
            ))
        
        def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
            a, b = (i1, j1), (i1, j2)
            c, d = (i2, j1), (i2, j2)
            e, f = (j1, k1), (j1, k2)
            g, h = (j2, k1), (j2, k2)
            C11, C12 = (i1, k1), (i1, k2)
            C21, C22 = (i2, k1), (i2, k2)
            
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        # Row 0: schoolbook
        for j in range(m):
            for k in range(p):
                add_schoolbook(0, j, k)
        
        # Rows 1-2: Strassen on 2x2 blocks
        for (j1, j2) in [(0, 1), (2, 3)]:
            for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
                add_strassen_2x2(1, 2, j1, j2, k1, k2)
        
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W

    def verify(U, V, W):
        import numpy as np
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_solve_for_W(U, V, target_rank):
        """
        Given U and V matrices of shape (n*m, target_rank) and (m*p, target_rank),
        solve for W of shape (p*n, target_rank) such that the tensor identity holds.
        
        The tensor identity: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        This is a system of n*m*p = 72 equations in target_rank * p*n unknowns.
        But it's nonlinear in W (each equation is linear in W, but the coefficients
        depend on U and V).
        
        Actually, for fixed U and V, each equation is linear in W:
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        This is linear in the W variables. So we can set up a linear system.
        """
        import numpy as np
        
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        rank = target_rank
        
        # Unknowns: W[k*n+i, r] for k in range(p), i in range(n), r in range(rank)
        # Total unknowns: p * n * rank
        
        # Equations: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Total equations: n * m * p = 72
        
        num_unknowns = p * n * rank
        num_equations = n * m * p
        
        # Build the linear system A * x = b
        # x is a vector of length num_unknowns
        # x[k*n*p + i*n + r] = W[k*n+i, r]  (flattened)
        
        # Actually, let's index x as x[(k, i, r)] = W[k*n+i, r]
        # x index = (k * n + i) * rank + r
        
        A = np.zeros((num_equations, num_unknowns), dtype=float)
        b = np.ones(num_equations, dtype=float)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        coeff = Ua[i * m + j, r] * Va[j * p + k, r]
                        if abs(coeff) > 1e-15:
                            x_idx = (k * n + i) * rank + r
                            A[eq_idx, x_idx] += coeff
                    eq_idx += 1
        
        # Solve the linear system
        try:
            x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Check residual
        residual = np.max(np.abs(A @ x - b))
        if residual > 1e-9:
            return None
        
        # Reconstruct W
        W = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W[k * n + i][r] = x[(k * n + i) * rank + r]
        
        return W

    def build_structured_UV(rank):
        """
        Build structured U and V matrices for a given rank.
        
        Strategy: Use the Strassen structure as a template.
        For rank 66, we know the exact U and V from the base decomposition.
        For lower rank, we need to find a valid structure.
        
        Let's try: for rank 60, use the same U and V as rank 66 but with 6 columns zeroed out.
        Then solve for W. If the system is consistent, we have a valid rank-60 decomposition.
        """
        import numpy as np
        
        # Start with the rank-66 U and V
        terms66 = build_rank66()
        U66, V66, W66 = terms_to_matrices(terms66)
        
        U66a = np.array(U66, dtype=float)
        V66a = np.array(V66, dtype=float)
        
        # Try different subsets of columns to keep
        # The 24 schoolbook terms (cols 0-23) are "pure" - each has one nonzero in U and V
        # The 42 Strassen terms (cols 24-65) have structure
        
        # Try keeping all 24 schoolbook terms and a subset of Strassen terms
        # We need to save 6 terms, so keep 36 of the 42 Strassen terms
        
        # Which Strassen terms to drop? Try different combinations.
        # The Strassen terms come in groups of 7 for each (j1,j2,k1,k2) block.
        # 6 blocks × 7 = 42 terms.
        
        # Try dropping 1 term from each of 6 blocks (6 terms total)
        # Or drop 2 terms from 3 blocks
        
        # Let's try: for each block, identify which term is most "redundant"
        
        # Actually, let's just try all subsets of size 6 to drop from the 42 Strassen terms
        # That's C(42,6) = 5,245,786 which is too many.
        
        # Instead, try: drop 1 term from each of 6 blocks
        # 7 choices per block, 6 blocks: 7^6 = 117,649 combinations
        
        # For each combination, check if the linear system for W is consistent
        
        # First, identify the Strassen term indices
        # Terms 0-23: schoolbook (row 0)
        # Terms 24-30: Strassen block (j=0,1; k=0,1)
        # Terms 31-37: Strassen block (j=0,1; k=2,3)
        # Terms 38-44: Strassen block (j=0,1; k=4,5)
        # Terms 45-51: Strassen block (j=2,3; k=0,1)
        # Terms 52-58: Strassen block (j=2,3; k=2,3)
        # Terms 59-65: Strassen block (j=2,3; k=4,5)
        
        strassen_blocks = [
            list(range(24, 31)),  # block 0
            list(range(31, 38)),  # block 1
            list(range(38, 45)),  # block 2
            list(range(45, 52)),  # block 3
            list(range(52, 59)),  # block 4
            list(range(59, 66)),  # block 5
        ]
        
        # Try dropping 1 term from each block
        from itertools import product
        
        for drop_combo in product(*[range(7) for _ in range(6)]):
            if not time_ok():
                break
            
            # drop_combo[r] = which term to drop from block r
            keep_indices = list(range(24))  # keep all schoolbook terms
            for block_idx, block in enumerate(strassen_blocks):
                for pos, term_idx in enumerate(block):
                    if pos != drop_combo[block_idx]:
                        keep_indices.append(term_idx)
            
            target_rank = len(keep_indices)
            if target_rank != 60:
                continue
            
            # Build U and V with only the kept columns
            U_sub = U66a[:, keep_indices]
            V_sub = V66a[:, keep_indices]
            
            # Solve for W
            W_sol = try_solve_for_W(U_sub, V_sub, target_rank)
            if W_sol is not None:
                # Found a valid rank-60 decomposition!
                return U_sub.tolist(), V_sub.tolist(), W_sol
        
        return None

    # Build base decomposition
    terms = build_rank66()
    U, V, W = terms_to_matrices(terms)
    
    # Try to find a lower-rank decomposition
    if time_ok():
        result = build_structured_UV(60)
        if result is not None:
            U, V, W = result
    
    # Verify
    if time_ok():
        if not verify(U, V, W):
            # Fall back to known good decomposition
            terms = build_rank66()
            U, V, W = terms_to_matrices(terms)
    
    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
