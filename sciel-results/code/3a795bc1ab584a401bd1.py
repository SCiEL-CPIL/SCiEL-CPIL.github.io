import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    def strassen_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def build_rank28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff

        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx

                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff

                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff

                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff

                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)

        return U, V, W

    def verify(U, V, W, A_test, B_test):
        C_expected = A_test @ B_test
        rank = U.shape[1]
        C_actual = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                C_actual[ii, kk] = acc
        return np.allclose(C_expected, C_actual, atol=1e-9)

    # Build the rank-28 solution
    U, V, W = build_rank28()

    # Tensor splitting approach: split m=4 into two halves of size 2
    # For each half h in {0,1}, we have a 2x2x4 sub-tensor
    # The idea: find rank-r decompositions for each 2x2x4 sub-tensor,
    # then try to merge them by sharing U-factors

    def build_2x2x4_decomposition(rank_target):
        """
        Build a rank-r decomposition for 2x2x4 matrix multiplication.
        A is 2x2, B is 2x4, C is 2x4.
        T[i,j,k] = sum_l A[i,l] * B[l,k] * C[k,i]
        """
        # For 2x2x4, the trivial decomposition has rank 8 (2*2*4/2... no, 2*2*4 = 16 entries)
        # Actually, rank of 2x2x4 multiplication tensor:
        # We can use Strassen-like ideas. Let's try rank 6, 7, 8.
        
        # Start with rank 8 (trivial):
        # U[i,l,s] for s in 0..7, V[l,k,s], W[k,i,s]
        # Trivial: s = l*4 + k, U[i,l,s] = delta(i,l), V[l,k,s] = delta(l,k), W[k,i,s] = 1
        # But that's rank 8 with many zeros.
        
        # Let's try to find a lower rank decomposition.
        # The tensor has 2*2*4 = 16 entries.
        # Lower bound: rank >= max(2*2, 2*4, 4*2) / something... 
        # Actually, for matrix multiplication [n,m,p], rank >= max(n*m, m*p, p*n) is NOT a lower bound.
        # The trivial rank is n*m*p.
        
        # Let's just try to find a rank-6 or rank-7 decomposition by optimization.
        # But for now, let's use rank 8 as a starting point and see if we can reduce.
        
        # Actually, let's think about this differently.
        # For 2x2x4, we can write:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
        
        # Trivial rank 8:
        U_sub = np.zeros((2*2, 8), dtype=float)  # (n*m, rank) = (4, 8)
        V_sub = np.zeros((2*4, 8), dtype=float)  # (m*p, rank) = (8, 8)
        W_sub = np.zeros((4*2, 8), dtype=float)  # (p*n, rank) = (8, 8)
        
        for l in range(2):
            for k in range(4):
                s = l * 4 + k
                U_sub[0*2 + l, s] = 1.0  # A[0,l]
                U_sub[1*2 + l, s] = 1.0  # A[1,l]
                V_sub[l*4 + k, s] = 1.0  # B[l,k]
                W_sub[k*2 + 0, s] = 1.0  # C[k,0]
                W_sub[k*2 + 1, s] = 1.0  # C[k,1]
        
        return U_sub, V_sub, W_sub

    # Try to find a lower-rank decomposition for 2x2x4
    # The key insight: for 2x2x4, we might be able to do better than rank 8.
    # Let's try rank 6 using a Strassen-like approach.
    
    def build_2x2x4_rank6():
        """
        Attempt a rank-6 decomposition for 2x2x4.
        Using Strassen-like ideas:
        Let A = [[a, b], [c, d]], B = [[e, f, g, h], [i, j, k, l]]
        C = [[a*e + b*i, a*f + b*j, a*g + b*k, a*h + b*l],
             [c*e + d*i, c*f + d*j, c*g + d*k, c*h + d*l]]
        
        Strassen-like:
        m1 = (a+b)*(e+i) = a*e + a*i + b*e + b*i
        m2 = a*(f-j) = a*f - a*j
        m3 = (c-d)*(i+l) = c*i + c*l - d*i - d*l
        m4 = d*(g-k) = d*g - d*k
        m5 = (a+c)*(e) = a*e + c*e
        m6 = (b+d)*(l) = b*l + d*l
        
        Then:
        C[0,0] = m1 - m3 - m5 + ... hmm, this is getting complicated.
        
        Let me try a different approach. For 2x2x4, the rank is at least 6 
        (by a counting argument or known results). Let's try to construct rank 6.
        """
        # Actually, let's just use the trivial rank-8 and see if the merging helps.
        # The total rank would be 8 + 8 = 16 if we can merge perfectly,
        # or up to 16 if we can't.
        # But the current rank is 28, so even rank 16 would be an improvement.
        
        # Wait, that's not right. Let me reconsider.
        # If we split B into B1 (rows 0-1) and B2 (rows 2-3), then:
        # C = A * B1 + A * B2 (where A*B1 is 2x2 and A*B2 is 2x2, concatenated to 2x4)
        # No, that's not how matrix multiplication works.
        
        # Actually, A is 2x4, B is 4x4.
        # A = [A1 | A2] where A1, A2 are 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = A1*B11 + A2*B21 (for left 2x2 of C)
        #   = A1*B12 + A2*B22 (for right 2x2 of C)
        
        # So we have 4 sub-products of 2x2*2x2, each of rank 7 (Strassen).
        # Total: 4*7 = 28. That's what we have now.
        
        # The tensor splitting idea: can we do better?
        # For the left half of C: C_left = A1*B11 + A2*B21
        # This is a 2x2 matrix that's a sum of two 2x2*2x2 products.
        # The tensor for this is: T_left[i,j,k] = sum_l (A1[i,l]*B11[l,k] + A2[i,l]*B21[l,k])
        # = sum_l A1[i,l]*B11[l,k] + sum_l A2[i,l]*B21[l,k]
        
        # This is the sum of two 2x2x2 tensors. The rank of a sum of two rank-r tensors
        # is at most 2r, but could be less if they share structure.
        
        # For 2x2x2, rank is 7 (Strassen). So the sum has rank at most 14.
        # Similarly for C_right, rank at most 14.
        # Total: at most 28. Same as before.
        
        # But wait! The key insight: A1 and A2 share the same A-indices.
        # Specifically, A1[i,l] and A2[i,l] both involve A[i, l] and A[i, l+2].
        # So the U-factors for the left and right halves might be mergeable.
        
        # Let me think about this more carefully.
        # For C_left[i,k] = sum_{l=0}^{1} A[i,l]*B[l,k] + sum_{l=0}^{1} A[i,l+2]*B[l+2,k]
        # The U-factor for term s in the first sum: U[i, s] involves A[i, l] for some l in {0,1}
        # The U-factor for term t in the second sum: U[i, t] involves A[i, l+2] for some l in {0,1}
        
        # These are different A-indices, so they can't be directly merged.
        
        # Hmm, so the tensor splitting doesn't obviously help here.
        
        # Let me try a different approach: look for a rank-27 decomposition directly.
        pass
        return None

    # Let's try a more direct approach: use the existing rank-28 decomposition
    # and try to find a linear dependency that allows reducing to rank 27.
    
    if time.time() > deadline - 30:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the tensor matrix M where M[:,s] = vec(U_s ⊗ V_s ⊗ W_s)
    n_terms = 28
    dim = n * m * m * p * p * n  # 2*4*4*4*4*2 = 1024... wait
    # Actually: i in range(n*m)=8, j in range(m*p)=16, k in range(p*n)=8
    # So dim = 8*16*8 = 1024
    
    dim = (n*m) * (m*p) * (p*n)  # 8 * 16 * 8 = 1024

    M = np.zeros((dim, n_terms), dtype=float)

    for s in range(n_terms):
        u_s = U[:, s]  # shape (n*m,) = (8,)
        v_s = V[:, s]  # shape (m*p,) = (16,)
        w_s = W[:, s]  # shape (p*n,) = (8,)

        for i in range(n * m):
            for j in range(m * p):
                base_idx = i * (m * p) * (p * n) + j * (p * n)
                M[base_idx:base_idx + p * n, s] = u_s[i] * v_s[j] * w_s

    if time.time() > deadline - 20:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check numerical rank
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
        if rank_M < 28 and time.time() < deadline - 15:
            # Find null space to try to eliminate a term
            _, s_vals, vh = np.linalg.svd(M, full_matrices=False)
            null_dim = 28 - rank_M

            A_test = np.arange(n * m).reshape(n, m).astype(float) + 1.0
            B_test = np.arange(m * p).reshape(m, p).astype(float) + 2.0

            for nv_idx in range(null_dim):
                if time.time() > deadline - 10:
                    break

                null_vec = vh[rank_M + nv_idx, :]
                max_idx = np.argmax(np.abs(null_vec))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue

                for remove_idx in range(n_terms):
                    if abs(null_vec[remove_idx]) < 1e-8:
                        continue

                    c_new = np.ones(n_terms) + null_vec
                    if abs(c_new[remove_idx]) > 1e-6:
                        continue

                    U_new = np.zeros((n * m, 27), dtype=float)
                    V_new = np.zeros((m * p, 27), dtype=float)
                    W_new = np.zeros((p * n, 27), dtype=float)
                    col_map = [s for s in range(n_terms) if s != remove_idx]
                    for new_col, old_col in enumerate(col_map):
                        U_new[:, new_col] = U[:, old_col] * c_new[old_col]
                        V_new[:, new_col] = V[:, old_col] * c_new[old_col]
                        W_new[:, new_col] = W[:, old_col] * c_new[old_col]

                    if verify(U_new, V_new, W_new, A_test, B_test):
                        U_out = np.round(U_new * 2) / 2
                        V_out = np.round(V_new * 2) / 2
                        W_out = np.round(W_new * 2) / 2

                        if (np.max(np.abs(U_new - U_out)) < 1e-9 and
                            np.max(np.abs(V_new - V_out)) < 1e-9 and
                            np.max(np.abs(W_new - W_out)) < 1e-9):
                            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                        else:
                            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    except Exception:
        pass

    # Try to find a rank-27 decomposition by solving for coefficients
    if time.time() < deadline - 10:
        try:
            for remove_idx in range(n_terms):
                if time.time() > deadline - 5:
                    break

                M_without = np.delete(M, remove_idx, axis=1)
                col_to_remove = M[:, remove_idx]

                try:
                    x, residuals, rank_r, svals = np.linalg.lstsq(M_without, col_to_remove, rcond=1e-10)
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-8:
                        c_new = np.ones(n_terms)
                        for new_idx, old_idx in enumerate([s for s in range(n_terms) if s != remove_idx]):
                            c_new[old_idx] = 1.0 + x[new_idx]
                        c_new[remove_idx] = 0.0

                        U_new = np.zeros((n * m, 27), dtype=float)
                        V_new = np.zeros((m * p, 27), dtype=float)
                        W_new = np.zeros((p * n, 27), dtype=float)
                        col_map = [s for s in range(n_terms) if s != remove_idx]
                        for new_col, old_col in enumerate(col_map):
                            U_new[:, new_col] = U[:, old_col] * c_new[old_col]
                            V_new[:, new_col] = V[:, old_col] * c_new[old_col]
                            W_new[:, new_col] = W[:, old_col] * c_new[old_col]

                        A_test = np.arange(n * m).reshape(n, m).astype(float) + 1.0
                        B_test = np.arange(m * p).reshape(m, p).astype(float) + 2.0

                        if verify(U_new, V_new, W_new, A_test, B_test):
                            U_out = np.round(U_new * 2) / 2
                            V_out = np.round(V_new * 2) / 2
                            W_out = np.round(W_new * 2) / 2

                            if (np.max(np.abs(U_new - U_out)) < 1e-9 and
                                np.max(np.abs(V_new - V_out)) < 1e-9 and
                                np.max(np.abs(W_new - W_out)) < 1e-9):
                                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                            else:
                                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                except Exception:
                    continue
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
