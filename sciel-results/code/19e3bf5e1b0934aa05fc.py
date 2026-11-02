"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 2, 2, 13

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based: 5 blocks of 2x2 (rank 7 each) + 1 block of 2x3 (rank 9)
        # Total: 5*7 + 9 = 44
        # Wait, let me recalculate: 13 = 2*6 + 1, so 6 blocks of 2 + 1 block of 1
        # Or: 13 = 2*5 + 3, so 5 blocks of 2 + 1 block of 3
        # 5*7 + 9 = 44
        
        # Actually, let me verify R(2x2x2x3) = 9
        # The 2x2x2x3 tensor: A(2x2) * B(2x3) = C(2x3)
        # Rank should be 9 based on known results
        
        rank_total = 5 * 7 + 9
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))

        # Strassen terms for 2x2x2x2
        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        # 5 blocks of 2 columns each
        for block in range(5):
            j0 = block * 2
            j1 = block * 2 + 1

            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1

        # Last block: 3 columns (j=10, 11, 12) with rank 9
        # 2x2x2x3 decomposition
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01, b02], [b10, b11, b12]]
        # C = [[c00, c01, c02], [c10, c11, c12]]
        # C indices in W: j*n+i, so c00=W[0], c10=W[1], c01=W[2], c11=W[3], c02=W[4], c12=W[5]
        
        # 9-term decomposition for 2x2x2x3:
        # Use Strassen on first 2 cols (terms 0-6) + 3 terms for col 2, but merge
        # Actually, let's use: 3 terms for col 0 (from Strassen-like), 3 terms for cols 1-2 (Strassen-like)
        
        # Better approach: 
        # Terms 0-2: handle column 0 of B (b00, b10) -> c00, c10
        # Terms 3-8: handle columns 1-2 of B (b01,b02,b11,b12) -> c01,c02,c11,c12
        
        # For col 0 (2x2x2x1): need 4 terms (schoolbook) or fewer
        # Actually 2x2x2x1 has rank 4 (can't do better)
        # For cols 1-2 (2x2x2x2): need 7 terms (Strassen)
        # Total: 4 + 7 = 11, not 9
        
        # So R(2x2x2x3) is actually 11, not 9!
        # Let me reconsider.
        
        # Actually, I think R(2x2x2x3) = 10 or 11.
        # Let me just use 5 blocks of 2 + 1 block of 3 with schoolbook for the 3-block
        # 5*7 + 6 = 41... no, 2x2x2x3 schoolbook is 2*2*3=12
        
        # Let me just stick with the proven 46: 6 blocks of 2 + 1 block of 1
        rank_total = 6 * 7 + 4
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1

            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1

        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1

        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0

        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))

        if error < 1e-10:
            best_rank = rank_total
            best_U, best_V, best_W = U, V, W

    except Exception:
        pass

    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
