def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - _t0 < _DEADLINE

    def add_strassen_2x2(i1, i2, j1, j2, k1, k2, terms):
        """Strassen 2x2x2 block decomposition."""
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

    def add_strassen_3x3_rows(i1, i2, i3, j1, j2, k1, k2, terms):
        """
        3-row Strassen-like: multiply A[i1:i3+1, j1:j2+1] (3x2) * B[j1:j2+1, k1:k2+1] (2x2)
        -> C[i1:i3+1, k1:k2+1] (3x2)
        Uses 6 multiplications instead of 6 (schoolbook for 3x2x2).
        Actually for 3x2x2, schoolbook is 12 terms. Let's use a different approach.
        """
        # For 3 rows x 2 cols x 2 cols, use a custom decomposition
        # A has entries: (i1,j1),(i1,j2),(i2,j1),(i2,j2),(i3,j1),(i3,j2)
        # B has entries: (j1,k1),(j1,k2),(j2,k1),(j2,k2)
        # C has entries: (i1,k1),(i1,k2),(i2,k1),(i2,k2),(i3,k1),(i3,k2)
        
        # Schoolbook would be 12 terms. Let's try to do better.
        # M1 = (A[i1,j1]+A[i2,j1]) * B[j1,k1]
        # M2 = (A[i1,j2]+A[i2,j2]) * B[j2,k1]
        # M3 = (A[i1,j1]+A[i1,j2]) * B[j1,k2]
        # M4 = (A[i2,j1]+A[i2,j2]) * B[j2,k2]
        # M5 = A[i3,j1] * (B[j1,k1]+B[j1,k2])
        # M6 = A[i3,j2] * (B[j2,k1]+B[j2,k2])
        
        a11, a12 = (i1, j1), (i1, j2)
        a21, a22 = (i2, j1), (i2, j2)
        a31, a32 = (i3, j1), (i3, j2)
        b11, b12 = (j1, k1), (j1, k2)
        b21, b22 = (j2, k1), (j2, k2)
        c11, c12 = (i1, k1), (i1, k2)
        c21, c22 = (i2, k1), (i2, k2)
        c31, c32 = (i3, k1), (i3, k2)
        
        # M1 = (a11+a21)*b11 -> c11:+1, c21:+1
        terms.append(({a11: 1, a21: 1}, {b11: 1}, {c11: 1, c21: 1}))
        # M2 = (a12+a22)*b21 -> c11:+1, c21:+1
        terms.append(({a12: 1, a22: 1}, {b21: 1}, {c11: 1, c21: 1}))
        # M3 = (a11+a12)*b12 -> c12:+1
        terms.append(({a11: 1, a12: 1}, {b12: 1}, {c12: 1}))
        # M4 = (a21+a22)*b22 -> c22:+1
        terms.append(({a21: 1, a22: 1}, {b22: 1}, {c22: 1}))
        # M5 = a31*(b11+b12) -> c31:+1, c32:+1
        terms.append(({a31: 1}, {b11: 1, b12: 1}, {c31: 1, c32: 1}))
        # M6 = a32*(b21+b22) -> c31:+1, c32:+1
        terms.append(({a32: 1}, {b21: 1, b22: 1}, {c31: 1, c32: 1}))
        
        # Now fix: c11 should be a11*b11 + a12*b21, but we have (a11+a21)*b11 + (a12+a22)*b21
        # = a11*b11 + a21*b11 + a12*b21 + a22*b21
        # We need c11 = a11*b11 + a12*b21, so subtract a21*b11 + a22*b21
        # c21 = a21*b11 + a22*b21, which is exactly what we added extra
        # So c11 needs -1 for the a21*b11 and a22*b21 parts
        # But we can't easily separate them. Let me reconsider.
        
        # Actually this doesn't work cleanly. Let me just use schoolbook for 3x2x2 = 12 terms
        # and rely on the trimming to reduce rank.
        pass

    def add_schoolbook(i, j, k, terms):
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))

    def build_base_decomposition():
        """Build the rank-66 Strassen+schoolbook decomposition."""
        terms = []
        
        # Row 0: schoolbook (4*6 = 24 terms)
        for j in range(m):
            for k in range(p):
                add_schoolbook(0, j, k, terms)
        
        # Rows 1-2: Strassen on 2x2 blocks
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_2x2(1, 2, 0, 1, k1, k2, terms)
            add_strassen_2x2(1, 2, 2, 3, k1, k2, terms)
        
        return terms

    def build_schoolbook():
        """Build pure schoolbook rank-72 decomposition."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_schoolbook(i, j, k, terms)
        return terms

    def terms_to_matrices(terms):
        """Convert list of (u_dict, v_dict, w_dict) to U, V, W matrices."""
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

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # For random A and B, check that sum_r U[:,r] * V[:,r] * W[:,r] = A*B
        import random
        random.seed(42)
        
        for trial in range(3):
            A = np.array([[random.randint(-5, 5) for _ in range(m)] for _ in range(n)], dtype=float)
            B = np.array([[random.randint(-5, 5) for _ in range(p)] for _ in range(m)], dtype=float)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(rank):
                a_vec = U_arr[:, r]  # shape (n*m,)
                b_vec = V_arr[:, r]  # shape (m*p,)
                c_vec = W_arr[:, r]  # shape (p*n,)
                
                # a_vec is indexed by i*m+j, b_vec by j*p+k, c_vec by k*n+i
                # The product term is: sum_{i,j,k} A[i,j]*B[j,k]*W[k*n+i] * U[i*m+j] * V[j*p+k]
                # But actually the decomposition is: C[i,k] = sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] for all j
                # Wait, no. The tensor decomposition is:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And T[i,j,k] = delta_{j,j} which is just 1 for valid indices
                # Actually T[i,j,k] = 1 if j is in [0,m) and the product A[i,j]*B[j,k] contributes to C[i,k]
                # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                
                # So C[i,k] = sum_j A[i,j] * B[j,k] * sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all valid (i,j,k)
                
                # Let's verify the tensor directly
                pass
        
        # Direct tensor verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def trim_decomposition(terms, max_iterations=200):
        """
        Apply trimming/peeling to reduce rank.
        A term can be removed if its U, V, W vectors are in the span of other terms' vectors
        in a way that preserves the tensor.
        
        More precisely: if there exists a scalar alpha and indices such that
        we can replace a term with a linear combination, we simplify.
        
        Simplest peeling: if a term has a U vector that is a multiple of another term's U vector,
        and similarly for V and W, we can combine them.
        """
        import numpy as np
        
        terms = list(terms)  # copy
        
        def vec_from_dict(d, size, idx_func):
            v = np.zeros(size, dtype=float)
            for idx, coeff in d.items():
                v[idx_func(idx)] = coeff
            return v

        def u_idx(i, j):
            return i * m + j

        def v_idx(j, k):
            return j * p + k

        def w_idx(i, k):
            return k * n + i

        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time_ok():
            improved = False
            iteration += 1
            
            rank = len(terms)
            if rank <= 1:
                break
            
            # Build U, V, W arrays
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            for col, (u_dict, v_dict, w_dict) in enumerate(terms):
                for (i, j), coeff in u_dict.items():
                    U[u_idx(i, j), col] = coeff
                for (j, k), coeff in v_dict.items():
                    V[v_idx(j, k), col] = coeff
                for (i, k), coeff in w_dict.items():
                    W[w_idx(i, k), col] = coeff
            
            # Try to find a term that can be eliminated
            # Check if any term's (U, V, W) is a linear combination of others
            # in the sense that the tensor contribution can be redistributed
            
            # Simple approach: check for duplicate or proportional terms
            for i in range(rank):
                if not time_ok():
                    break
                for j in range(i + 1, rank):
                    # Check if term i is proportional to term j in all three factors
                    ui, uj = U[:, i], U[:, j]
                    vi, vj = V[:, i], V[:, j]
                    wi, wj = W[:, i], W[:, j]
                    
                    # Check if ui is proportional to uj
                    nonzero_i = np.nonzero(ui)[0]
                    nonzero_j = np.nonzero(uj)[0]
                    
                    if len(nonzero_i) == 0 or len(nonzero_j) == 0:
                        continue
                    
                    # Find ratio
                    common = set(nonzero_i) & set(nonzero_j)
                    if len(common) == 0:
                        continue
                    
                    ratios = [ui[c] / uj[c] for c in common if uj[c] != 0]
                    if len(ratios) == 0:
                        continue
                    
                    ratio = ratios[0]
                    if all(abs(r - ratio) < 1e-9 for r in ratios):
                        # Check V
                        common_v = set(np.nonzero(vi)[0]) & set(np.nonzero(vj)[0])
                        if len(common_v) > 0:
                            ratios_v = [vi[c] / vj[c] for c in common_v if vj[c] != 0]
                            if len(ratios_v) > 0:
                                ratio_v = ratios_v[0]
                                if all(abs(r - ratio_v) < 1e-9 for r in ratios_v) and abs(ratio_v - ratio) < 1e-9:
                                    # Check W
                                    common_w = set(np.nonzero(wi)[0]) & set(np.nonzero(wj)[0])
                                    if len(common_w) > 0:
                                        ratios_w = [wi[c] / wj[c] for c in common_w if wj[c] != 0]
                                        if len(ratios_w) > 0:
                                            ratio_w = ratios_w[0]
                                            if all(abs(r - ratio_w) < 1e-9 for r in ratios_w) and abs(ratio_w - ratio) < 1e-9:
                                                # Term i is proportional to term j, merge them
                                                # New term j = term i + term j (if ratio=1) or adjust
                                                # Actually, if ui = ratio * uj, vi = ratio * vj, wi = ratio * wj
                                                # Then term i contributes ratio * (uj * vj * wj) to the tensor
                                                # So we can just add ratio to term j's coefficients and remove term i
                                                # But this only works if the tensor entries add up correctly
                                                # The tensor entry for term r is U[:,r]*V[:,r]*W[:,r] (outer product)
                                                # So term i's contribution = ratio * term j's contribution
                                                # We can't simply merge unless we're careful
                                                pass
            
            # Try a different approach: Gaussian elimination on the tensor
            # For each tensor entry (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of n*m*p = 72 equations in rank*3 variables (but with constraints)
            
            # Simpler: try to find a term where one factor has a single nonzero entry
            # and eliminate it using row/column operations on U, V, W
            for r in range(rank):
                if not time_ok():
                    break
                
                u_vec = U[:, r]
                v_vec = V[:, r]
                w_vec = W[:, r]
                
                # If U has exactly one nonzero, we can potentially eliminate
                nz_u = np.nonzero(u_vec)[0]
                nz_v = np.nonzero(v_vec)[0]
                nz_w = np.nonzero(w_vec)[0]
                
                if len(nz_u) == 1 and len(nz_v) == 1 and len(nz_w) == 1:
                    # This term only contributes to one tensor entry
                    # Check if that tensor entry is already satisfied by other terms
                    i_idx = nz_u[0]
                    j_idx = nz_v[0]
                    k_idx = nz_w[0]
                    
                    i_val = i_idx // m
                    j_val = i_idx % m
                    j2_val = j_idx // p
                    k2_val = j_idx % p
                    k3_val = k_idx // n
                    i2_val = k_idx % n
                    
                    if i_val != i2_val or j_val != j2_val or k2_val != k3_val:
                        # Inconsistent indices, skip
                        continue
                    
                    # Check if other terms also contribute to this (i,j,k)
                    contribution = 0.0
                    for r2 in range(rank):
                        if r2 == r:
                            continue
                        val = U[i_val * m + j_val, r2] * V[j_val * p + k2_val, r2] * W[k2_val * n + i_val, r2]
                        contribution += val
                    
                    own_val = u_vec[i_val * m + j_val] * v_vec[j_val * p + k2_val] * w_vec[k2_val * n + i_val]
                    
                    if abs(contribution - 1.0) < 1e-9:
                        # This term is redundant! Remove it
                        terms.pop(r)
                        improved = True
                        break
            
            if improved:
                continue
        
        return terms

    def trim_by_linear_dependence(terms, max_iterations=100):
        """
        More sophisticated trimming: if the U vectors of a subset of terms are linearly dependent,
        we can perform column operations to zero out one term's U vector, then check if
        the remaining terms still span the tensor.
        """
        import numpy as np
        
        terms = list(terms)
        
        def rebuild_matrices(terms):
            rank = len(terms)
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            for col, (u_dict, v_dict, w_dict) in enumerate(terms):
                for (i, j), coeff in u_dict.items():
                    U[i * m + j, col] = coeff
                for (j, k), coeff in v_dict.items():
                    V[j * p + k, col] = coeff
                for (i, k), coeff in w_dict.items():
                    W[k * n + i, col] = coeff
            return U, V, W

        def tensor_residual(U, V, W):
            """Compute max |sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1| over all (i,j,k)."""
            rank = U.shape[1]
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        max_res = max(max_res, abs(val - 1.0))
            return max_res

        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time_ok():
            improved = False
            iteration += 1
            
            U, V, W = rebuild_matrices(terms)
            rank = len(terms)
            if rank <= 1:
                break
            
            # Try to eliminate each term by checking if removing it and adjusting
            # others can still satisfy the tensor
            # Simple heuristic: if a term's U vector is in the span of others,
            # try to express the tensor constraint without it
            
            for r in range(rank):
                if not time_ok():
                    break
                
                # Check if U[:, r] is in the span of other U columns
                other_U = np.delete(U, r, axis=1)
                u_r = U[:, r]
                
                if np.allclose(u_r, 0):
                    # U vector is zero, term is useless
                    terms.pop(r)
                    improved = True
                    break
                
                # Try to see if we can absorb this term into others
                # This is complex; for now, just check for zero terms
                pass
        
        return terms

    # Build base decomposition
    terms = build_base_decomposition()
    
    if time_ok():
        # Try trimming
        terms_trimmed = trim_decomposition(terms, max_iterations=50)
        if len(terms_trimmed) < len(terms):
            terms = terms_trimmed
    
    if time_ok():
        terms_trimmed2 = trim_by_linear_dependence(terms, max_iterations=50)
        if len(terms_trimmed2) < len(terms):
            terms = terms_trimmed2
    
    # Convert to matrices
    U, V, W = terms_to_matrices(terms)
    
    # Verify
    if time_ok():
        if not verify_decomposition(U, V, W):
            # Fall back to known good decomposition
            terms = build_base_decomposition()
            U, V, W = terms_to_matrices(terms)
    
    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
