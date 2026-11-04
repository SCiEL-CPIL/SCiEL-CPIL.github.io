"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 3, 4, 6

    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    # Build the decomposition
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-8)

    def verify_fast(U_mat, V_mat, W_mat, r):
        # Faster verification using matrix multiplication structure
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can check this by building the tensor and comparing
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        return np.max(np.abs(T_reconstructed - 1.0)) < 1e-7

    if not verify(U, V, W, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    nmp = n * m * p

    def build_tensor_matrix(U_cur, V_cur, W_cur, r_cur):
        u = U_cur.reshape(n, m, r_cur)
        v = V_cur.reshape(m, p, r_cur)
        w = W_cur.reshape(p, n, r_cur)
        M = np.einsum('ijc,jkc,kic->ijkc', u, v, w).reshape(nmp, r_cur)
        return M

    def try_reduce_rank(U_cur, V_cur, W_cur, r_cur, max_reductions=30):
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        for reduction in range(max_reductions):
            if time.time() > _deadline - 20:
                break
                
            r = best_rank
            M = build_tensor_matrix(best_U, best_V, best_W, r)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-9
                rank_M = np.sum(s > tol)
                
                if rank_M >= r:
                    # No null space - try greedy deletion
                    found = False
                    for col_idx in range(r):
                        if time.time() > _deadline - 20:
                            break
                        U_new = np.delete(best_U, col_idx, axis=1)
                        V_new = np.delete(best_V, col_idx, axis=1)
                        W_new = np.delete(best_W, col_idx, axis=1)
                        if verify_fast(U_new, V_new, W_new, r - 1):
                            best_rank = r - 1
                            best_U = U_new
                            best_V = V_new
                            best_W = W_new
                            found = True
                            break
                    if not found:
                        break
                    continue
                
                null_space = Vt[rank_M:, :].T
                found = False
                
                for nv in null_space:
                    if time.time() > _deadline - 20:
                        break
                    
                    abs_nv = np.abs(nv)
                    significant = np.where(abs_nv > tol * 10)[0]
                    significant = significant[np.argsort(abs_nv[significant])[::-1]]
                    
                    # Strategy 1: Try removing the column with largest nullspace weight
                    for max_idx in significant[:min(8, len(significant))]:
                        if time.time() > _deadline - 20:
                            break
                        
                        U_new = np.delete(best_U, max_idx, axis=1)
                        V_new = np.delete(best_V, max_idx, axis=1)
                        W_new = np.delete(best_W, max_idx, axis=1)
                        
                        if verify_fast(U_new, V_new, W_new, r - 1):
                            best_rank = r - 1
                            best_U = U_new
                            best_V = V_new
                            best_W = W_new
                            found = True
                            break
                    
                    if found:
                        break
                    
                    # Strategy 2: Use nullspace vector to create linear combination
                    # If nv is in null space, then sum_c nv[c] * (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) = 0
                    # This means we can eliminate one column by combining others
                    if len(significant) >= 2:
                        # Find the column with smallest |nv[c]| that we can eliminate
                        # and express it as combination of others
                        for elim_idx in significant[::-1][:min(5, len(significant))]:
                            if time.time() > _deadline - 20:
                                break
                            if abs(nv[elim_idx]) < tol:
                                continue
                            
                            # Try to find a simple combination
                            # For each other column, check if we can merge
                            for other_idx in significant:
                                if other_idx == elim_idx:
                                    continue
                                if abs(nv[other_idx]) < tol:
                                    continue
                                
                                scale = -nv[elim_idx] / nv[other_idx]
                                U_new = best_U.copy()
                                V_new = best_V.copy()
                                W_new = best_W.copy()
                                U_new[:, other_idx] = U_new[:, other_idx] + scale * U_new[:, elim_idx]
                                V_new[:, other_idx] = V_new[:, other_idx] + scale * V_new[:, elim_idx]
                                W_new[:, other_idx] = W_new[:, other_idx] + scale * W_new[:, elim_idx]
                                # Remove elim_idx
                                U_new = np.delete(U_new, elim_idx, axis=1)
                                V_new = np.delete(V_new, elim_idx, axis=1)
                                W_new = np.delete(W_new, elim_idx, axis=1)
                                
                                if verify_fast(U_new, V_new, W_new, r - 1):
                                    best_rank = r - 1
                                    best_U = U_new
                                    best_V = V_new
                                    best_W = W_new
                                    found = True
                                    break
                            if found:
                                break
                    
                    if found:
                        break
                
                if not found:
                    # Try greedy deletion as fallback
                    for col_idx in range(r):
                        if time.time() > _deadline - 20:
                            break
                        U_new = np.delete(best_U, col_idx, axis=1)
                        V_new = np.delete(best_V, col_idx, axis=1)
                        W_new = np.delete(best_W, col_idx, axis=1)
                        if verify_fast(U_new, V_new, W_new, r - 1):
                            best_rank = r - 1
                            best_U = U_new
                            best_V = V_new
                            best_W = W_new
                            found = True
                            break
                    if not found:
                        break
                    
            except:
                break
        
        return best_rank, best_U, best_V, best_W

    best_rank, best_U, best_V, best_W = try_reduce_rank(U, V, W, 66, max_reductions=20)

    # Final polish: try to reduce further with a more aggressive search
    if time.time() < _deadline - 25:
        r2 = best_rank
        # Try removing individual columns one by one from the current best
        for attempt in range(r2):
            if time.time() > _deadline - 15:
                break
            found = False
            for col_idx in range(r2):
                if time.time() > _deadline - 15:
                    break
                U_new = np.delete(best_U, col_idx, axis=1)
                V_new = np.delete(best_V, col_idx, axis=1)
                W_new = np.delete(best_W, col_idx, axis=1)
                if verify_fast(U_new, V_new, W_new, r2 - 1):
                    best_rank = r2 - 1
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    r2 -= 1
                    found = True
                    break
            if not found:
                break
        
        # Try SVD-based reduction one more time with tighter tolerance
        if time.time() < _deadline - 20:
            r3 = best_rank
            M = build_tensor_matrix(best_U, best_V, best_W, r3)
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol_tight = 1e-8
                rank_M = np.sum(s > tol_tight)
                if rank_M < r3:
                    null_space = Vt[rank_M:, :].T
                    for nv in null_space:
                        if time.time() > _deadline - 15:
                            break
                        abs_nv = np.abs(nv)
                        significant = np.where(abs_nv > tol_tight)[0]
                        significant = significant[np.argsort(abs_nv[significant])[::-1]]
                        for max_idx in significant[:min(10, len(significant))]:
                            if time.time() > _deadline - 15:
                                break
                            U_new = np.delete(best_U, max_idx, axis=1)
                            V_new = np.delete(best_V, max_idx, axis=1)
                            W_new = np.delete(best_W, max_idx, axis=1)
                            if verify_fast(U_new, V_new, W_new, r3 - 1):
                                best_rank = r3 - 1
                                best_U = U_new
                                best_V = V_new
                                best_W = W_new
                                r3 -= 1
                                break
            except:
                pass

    if best_rank < 66:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
