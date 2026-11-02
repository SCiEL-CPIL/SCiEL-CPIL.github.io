def construct():
    import time
    import numpy as np
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def build_strassen_hybrid():
        """Build the hybrid Strassen+schoolbook decomposition (rank 66)."""
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        return terms

    def build_full_schoolbook():
        """Build the full schoolbook decomposition (rank 72)."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U @ diag(1) @ V^T contracted with W gives the multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check a few random entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            expected = 1.0  # For multiplication tensor, T[i,j,k]=1 if valid, else 0
            # Actually for matrix mult tensor, T[i,j,k] = 1 for all valid (i,j,k)
            # Wait no - the tensor is T[i,j,k] = delta_{i,j,k} meaning it's 1 when indices align
            # For C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] = 1 for all i,j,k (it's the identity mapping in the tensor sense)
            # Actually no. The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
            # because C[i,k] = sum_j A[i,j] B[j,k], so each triple (i,j,k) contributes 1*A[i,j]*B[j,k]
            if abs(val - 1.0) > tol:
                return False
        return True

    def verify_full(U, V, W, tol=1e-9):
        """Full verification of the decomposition."""
        # T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def reduce_rank(U, V, W, max_iter=2000):
        """
        Try to reduce rank by finding linear dependencies.
        Strategy: for each column r, check if U[:,r] can be expressed as combination of other U columns,
        and if so, adjust V and W to eliminate it.
        
        More systematic approach: use the fact that if we have terms t1 and t2 where
        U[:,r1] = c * U[:,r2], then we can eliminate one.
        
        Better: Gaussian elimination on the U matrix to find pivot columns,
        then for non-pivot columns, express them as linear combos and absorb.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        
        # Try iterative elimination: find pairs of columns where one is a scalar multiple
        # of the other in U (or V or W), and eliminate.
        
        eliminated = 0
        active = list(range(rank))
        
        # Build tracking: for each active column, we track its contribution
        # We'll work with the full matrices and zero out columns as we eliminate them.
        
        for iteration in range(max_iter):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            if len(active) < 2:
                break
            
            found = False
            for idx_a in range(len(active)):
                for idx_b in range(idx_a + 1, len(active)):
                    ra = active[idx_a]
                    rb = active[idx_b]
                    
                    # Check if U[:,ra] is a scalar multiple of U[:,rb]
                    ua = U[:, ra]
                    ub = U[:, rb]
                    
                    # Find non-zero entries
                    nz_a = np.nonzero(ua)[0]
                    nz_b = np.nonzero(ub)[0]
                    
                    if len(nz_a) == 0 or len(nz_b) == 0:
                        continue
                    
                    # Check if one is scalar multiple of other
                    # ua = c * ub
                    # Use first non-zero of ub
                    if abs(ub[nz_b[0]]) > 1e-12:
                        c = ua[nz_b[0]] / ub[nz_b[0]] if nz_b[0] in nz_a else None
                        if c is not None and abs(c) > 1e-12:
                            # Check if ua = c * ub
                            if np.allclose(ua, c * ub, atol=1e-10):
                                # Now check if V[:,ra] = c' * V[:,rb] or W[:,ra] = c'' * W[:,rb]
                                # For elimination: if U[:,ra] = c*U[:,rb], V[:,ra] = d*V[:,rb], W[:,ra] = e*W[:,rb]
                                # with c*d*e = 1, we can eliminate ra.
                                # But we need c*d*e = 1 for the product to work out.
                                
                                va = V[:, ra]
                                vb = V[:, rb]
                                nz_va = np.nonzero(va)[0]
                                nz_vb = np.nonzero(vb)[0]
                                
                                if len(nz_va) == 0 or len(nz_vb) == 0:
                                    continue
                                
                                # Check V relationship
                                if abs(vb[nz_vb[0]]) > 1e-12:
                                    d = va[nz_vb[0]] / vb[nz_vb[0]] if nz_vb[0] in nz_va else None
                                    if d is not None and abs(d) > 1e-12:
                                        if np.allclose(va, d * vb, atol=1e-10):
                                            # Check W
                                            wa = W[:, ra]
                                            wb = W[:, rb]
                                            nz_wa = np.nonzero(wa)[0]
                                            nz_wb = np.nonzero(wb)[0]
                                            
                                            if len(nz_wa) == 0 or len(nz_wb) == 0:
                                                continue
                                            
                                            if abs(wb[nz_wb[0]]) > 1e-12:
                                                e = wa[nz_wb[0]] / wb[nz_wb[0]] if nz_wb[0] in nz_wa else None
                                                if e is not None and abs(e) > 1e-12:
                                                    if np.allclose(wa, e * wb, atol=1e-10):
                                                        # Check c*d*e ≈ 1
                                                        if abs(c * d * e - 1.0) < 1e-8:
                                                            # Eliminate column ra
                                                            U[:, ra] = 0
                                                            V[:, ra] = 0
                                                            W[:, ra] = 0
                                                            active.remove(ra)
                                                            eliminated += 1
                                                            found = True
                                                            break
                if found:
                    break
            if not found:
                break
        
        return U, V, W, eliminated

    def reduce_rank_v2(U, V, W, max_iter=5000):
        """
        More general reduction: for any pair of active columns (ra, rb),
        check if there exist scalars c, d, e with c*d*e=1 such that
        U[:,ra] = c*U[:,rb], V[:,ra] = d*V[:,rb], W[:,ra] = e*W[:,rb].
        
        Actually, a better approach: use the tensor structure.
        If we have two terms where the U vectors are proportional AND the V vectors are proportional,
        then the W vectors must also be proportional (with the right scaling) for the tensor to be valid.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        active = set(range(rank))
        eliminated = 0
        
        for iteration in range(max_iter):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            if len(active) < 2:
                break
            
            found = False
            active_list = sorted(active)
            for idx_a in range(len(active_list)):
                for idx_b in range(idx_a + 1, len(active_list)):
                    ra = active_list[idx_a]
                    rb = active_list[idx_b]
                    
                    ua, ub = U[:, ra], U[:, rb]
                    va, vb = V[:, ra], V[:, rb]
                    wa, wb = W[:, ra], W[:, rb]
                    
                    # Check U proportionality
                    nz_ub = np.nonzero(ub)[0]
                    nz_ua = np.nonzero(ua)[0]
                    if len(nz_ub) == 0 or len(nz_ua) == 0:
                        continue
                    
                    # Try to find c such that ua ≈ c*ub
                    c_found = False
                    for pos in nz_ub:
                        if abs(ub[pos]) > 1e-12:
                            c = ua[pos] / ub[pos] if ua[pos] != 0 or abs(ub[pos]) > 1e-12 else 0
                            # Check all positions
                            if np.allclose(ua, c * ub, atol=1e-10):
                                c_found = True
                                break
                    if not c_found or abs(c) < 1e-12:
                        continue
                    
                    # Check V proportionality
                    nz_vb = np.nonzero(vb)[0]
                    nz_va = np.nonzero(va)[0]
                    if len(nz_vb) == 0 or len(nz_va) == 0:
                        continue
                    
                    d_found = False
                    for pos in nz_vb:
                        if abs(vb[pos]) > 1e-12:
                            d = va[pos] / vb[pos] if va[pos] != 0 or abs(vb[pos]) > 1e-12 else 0
                            if np.allclose(va, d * vb, atol=1e-10):
                                d_found = True
                                break
                    if not d_found or abs(d) < 1e-12:
                        continue
                    
                    # Check W proportionality with required e = 1/(c*d)
                    e_needed = 1.0 / (c * d)
                    nz_wb = np.nonzero(wb)[0]
                    nz_wa = np.nonzero(wa)[0]
                    if len(nz_wb) == 0 or len(nz_wa) == 0:
                        continue
                    
                    if np.allclose(wa, e_needed * wb, atol=1e-10):
                        # Eliminate ra
                        U[:, ra] = 0
                        V[:, ra] = 0
                        W[:, ra] = 0
                        active.discard(ra)
                        eliminated += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        return U, V, W, eliminated

    # Start with the hybrid (rank 66)
    terms = build_strassen_hybrid()
    U, V, W = terms_to_matrices(terms)
    base_rank = U.shape[1]
    
    # Verify
    if not verify_full(U, V, W):
        # Fallback to schoolbook
        terms = build_full_schoolbook()
        U, V, W = terms_to_matrices(terms)
        base_rank = U.shape[1]
    
    # Try to reduce rank
    U, V, W, elim = reduce_rank_v2(U, V, W)
    
    # Verify after reduction
    if not verify_full(U, V, W, tol=1e-8):
        # Reduction failed, use original
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
    
    # Convert to lists with integer/half-integer entries
    def to_clean_list(mat):
        rows = []
        for row in mat:
            clean_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                if abs(rounded) < 1e-10:
                    rounded = 0.0
                clean_row.append(rounded)
            rows.append(clean_row)
        return rows
    
    U_list = to_clean_list(U)
    V_list = to_clean_list(V)
    W_list = to_clean_list(W)
    
    # Final verification with clean values
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        # Use the hybrid without reduction
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)
    
    # Remove all-zero columns to get true rank
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    nonzero_cols = []
    for c in range(U_np.shape[1]):
        if np.any(U_np[:, c] != 0) or np.any(V_np[:, c] != 0) or np.any(W_np[:, c] != 0):
            nonzero_cols.append(c)
    
    U_np = U_np[:, nonzero_cols]
    V_np = V_np[:, nonzero_cols]
    W_np = W_np[:, nonzero_cols]
    
    # Final check
    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)
    else:
        U_list = to_clean_list(U_np)
        V_list = to_clean_list(V_np)
        W_list = to_clean_list(W_np)
    
    return {"U": U_list, "V": V_list, "W": W_list}
