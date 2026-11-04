"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 145
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

    # Precompute indices for fast verification
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For fast check: reshape and use einsum efficiently
    def verify_fast(U_mat, V_mat, W_mat, r):
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape: u[i,j,c], v[j,k,c], w[k,i,c]
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        # Compute tensor
        T = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        return np.max(np.abs(T - 1.0)) < 1e-7

    if not verify_fast(U, V, W, 66):
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

    def try_reduce_rank(U_cur, V_cur, W_cur, r_cur, max_reductions=50):
        best_rank = r_cur
        best_U = U_cur.copy()
        best_V = V_cur.copy()
        best_W = W_cur.copy()
        
        for reduction in range(max_reductions):
            if time.time() > _deadline - 5:
                break
                
            r = best_rank
            
            # Greedy deletion: try removing each column
            found = False
            for col_idx in range(r):
                if time.time() > _deadline - 5:
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
        
        return best_rank, best_U, best_V, best_W

    best_rank, best_U, best_V, best_W = try_reduce_rank(U, V, W, 66, max_reductions=50)

    # Try SVD-based approach to find linear dependencies
    if time.time() < _deadline - 10 and best_rank < 66:
        r = best_rank
        # Build the "tensor matrix" where each column is the rank-1 tensor flattened
        # M[ijk, c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        u = best_U.reshape(n, m, r)
        v = best_V.reshape(m, p, r)
        w = best_W.reshape(p, n, r)
        # M has shape (n*m*p, r)
        M = np.einsum('ijc,jkc,kic->ijkc', u, v, w).reshape(n*m*p, r)
        
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9
            rank_M = int(np.sum(s > tol))
            
            if rank_M < r:
                # There are linear dependencies
                null_space = Vt[rank_M:, :].T
                # Try to use null space vectors to eliminate columns
                for nv in null_space:
                    if time.time() > _deadline - 5:
                        break
                    abs_nv = np.abs(nv)
                    # Try removing the column with largest weight
                    max_idx = int(np.argmax(abs_nv))
                    U_new = np.delete(best_U, max_idx, axis=1)
                    V_new = np.delete(best_V, max_idx, axis=1)
                    W_new = np.delete(best_W, max_idx, axis=1)
                    if verify_fast(U_new, V_new, W_new, r - 1):
                        best_rank = r - 1
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        r -= 1
        except:
            pass

    # Final aggressive reduction pass
    if time.time() < _deadline - 5:
        r2 = best_rank
        for attempt in range(r2):
            if time.time() > _deadline - 3:
                break
            found = False
            for col_idx in range(r2):
                if time.time() > _deadline - 3:
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

    if best_rank < 66:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
