"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build a block-structured decomposition:
    # - Split columns into 3 groups of 2
    # - For each group, use Strassen for the 2x2 sub-blocks in rows 1-2
    # - For row 0, use direct terms
    # This gives rank 66. Then attempt to reduce to rank 65 by finding
    # a linear dependence among the rank-1 terms.

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

        # Direct: A0*B00 -> C0, A1*B10 -> C0
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        # Direct: A0*B01 -> C1, A1*B11 -> C1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    # Build the decomposition
    # Row 0: direct terms for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    # Now attempt to reduce rank from 66 to 65
    # by finding a linear dependence among the columns.
    # The tensor decomposition is: sum_l U[:,l] * V[:,l] * W[:,l] = T
    # If we can find coefficients alpha_l such that sum_l alpha_l * (U[:,l] * V[:,l] * W[:,l]) = 0
    # and alpha is not all zero, then we can eliminate one term.

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
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

    # Try to find a rank-65 decomposition by eliminating one column.
    # For each candidate column c, check if T_c is in the span of the other 65 tensors.
    # Build the 72x66 tensor matrix and check column dependencies.
    
    nmp = n * m * p  # 72
    M = np.zeros((nmp, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, l] = U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]

    best_rank = 66
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to eliminate one column at a time
    # For each column c, check if removing it still gives a valid decomposition
    # by verifying the remaining 65 terms sum to the target tensor.
    # This is equivalent to checking if T_c is in the span of the others.
    
    # More efficient: use QR/SVD to find the nullspace of M
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Find the rank of M
        tol = 1e-10
        rank_M = np.sum(s > tol)
        
        if rank_M < 66:
            # There are (66 - rank_M) nullspace vectors
            # Each nullspace vector gives a linear dependence
            # Try to find a dependence where we can eliminate one term
            null_space = Vt[rank_M:, :].T  # (66, 66-rank_M)
            
            for nv in null_space:
                # Find the column with the largest absolute coefficient
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) < tol:
                    continue
                # Try eliminating this column
                U65 = np.delete(U, max_idx, axis=1)
                V65 = np.delete(V, max_idx, axis=1)
                W65 = np.delete(W, max_idx, axis=1)
                
                if verify(U65, V65, W65, 65):
                    best_rank = 65
                    best_result = {"U": U65.tolist(), "V": V65.tolist(), "W": W65.tolist()}
                    break
    except:
        pass
    
    # If we found rank 65, try to go further to rank 64
    if best_rank == 65:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        r_cur = 65
        
        M2 = np.zeros((nmp, r_cur))
        for l in range(r_cur):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M2[i * m * p + j * p + k, l] = U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
        
        try:
            U_svd2, s2, Vt2 = np.linalg.svd(M2, full_matrices=False)
            tol2 = 1e-10
            rank_M2 = np.sum(s2 > tol2)
            
            if rank_M2 < r_cur:
                null_space2 = Vt2[rank_M2:, :].T
                for nv in null_space2:
                    max_idx = np.argmax(np.abs(nv))
                    if abs(nv[max_idx]) < tol2:
                        continue
                    U64 = np.delete(U_cur, max_idx, axis=1)
                    V64 = np.delete(V_cur, max_idx, axis=1)
                    W64 = np.delete(W_cur, max_idx, axis=1)
                    
                    if verify(U64, V64, W64, 64):
                        best_rank = 64
                        best_result = {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
                        break
        except:
            pass
    
    # Additional search: try to reduce further using iterative nullspace search
    # with a time budget
    if best_rank > 64 and time.time() < _deadline - 20:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        r_cur = best_rank
        
        # Try multiple rounds of elimination
        for round_num in range(3):
            if time.time() > _deadline - 10:
                break
            
            M_cur = np.zeros((nmp, r_cur))
            for l in range(r_cur):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M_cur[i * m * p + j * p + k, l] = U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
            
            try:
                U_svd_r, s_r, Vt_r = np.linalg.svd(M_cur, full_matrices=False)
                tol_r = 1e-10
                rank_Mr = np.sum(s_r > tol_r)
                
                if rank_Mr < r_cur:
                    null_space_r = Vt_r[rank_Mr:, :].T
                    for nv in null_space_r:
                        if time.time() > _deadline - 10:
                            break
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < tol_r:
                            continue
                        U_new = np.delete(U_cur, max_idx, axis=1)
                        V_new = np.delete(V_cur, max_idx, axis=1)
                        W_new = np.delete(W_cur, max_idx, axis=1)
                        
                        if verify(U_new, V_new, W_new, r_cur - 1):
                            U_cur = U_new
                            V_cur = V_new
                            W_cur = W_new
                            r_cur = r_cur - 1
                            best_rank = r_cur
                            best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
                            break
            except:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
