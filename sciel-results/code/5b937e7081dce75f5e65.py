"""Matrix multiplication 4,4,9: tensor decomposition with rank minimization."""
DIMENSIONS = [4, 4, 9]

import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Schoolbook decomposition: rank = n*m*p = 144
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_rank = rank_school
    best_result = {"U": U, "V": V, "W": W}

    import numpy as np

    def verify_numpy(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Fast verification using NumPy einsum."""
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Build expected tensor
        T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_expected[a, b, c] = 1.0
        
        if np.max(np.abs(T - T_expected)) > tol:
            return False
        return True

    # Strassen 2x2x2 rank-7 decomposition coefficients
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def build_4x4x4_strassen():
        """Build rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2x2 blocks.
        Returns (U_cols, V_cols, W_cols) each 16x56 lists.
        """
        U_cols = [[0] * 56 for _ in range(16)]
        V_cols = [[0] * 56 for _ in range(16)]
        W_cols = [[0] * 56 for _ in range(16)]

        products = [
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
            (1, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ]

        for prod_idx, (ar, ac, br, bc) in enumerate(products):
            for s in range(7):
                col = prod_idx * 7 + s
                for i in range(2):
                    for j in range(2):
                        U_cols[(ar * 2 + i) * 4 + (ac * 2 + j)][col] = strassen_U[s][i * 2 + j]
                        V_cols[(br * 2 + i) * 4 + (bc * 2 + j)][col] = strassen_V[s][i * 2 + j]
                        W_cols[(bc * 2 + j) * 4 + (ar * 2 + i)][col] = strassen_W[s][i * 2 + j]

        return U_cols, V_cols, W_cols

    # Build rank-128 decomposition: 56 + 56 + 16
    if time.time() < deadline - 30:
        total_rank = 128
        U_new = [[0] * total_rank for _ in range(16)]
        V_new = [[0] * total_rank for _ in range(36)]
        W_new = [[0] * total_rank for _ in range(36)]

        # Block 1: cols 0-3 of B (4x4 sub-problem), rank 56
        U1, V1, W1 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new[a][r] = U1[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new[j * 9 + k][r] = V1[b][r]
            for c in range(16):
                W_new[c][r] = W1[c][r]

        # Block 2: cols 4-7 of B (4x4 sub-problem), rank 56
        U2, V2, W2 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new[a][56 + r] = U2[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new[j * 9 + (4 + k)][56 + r] = V2[b][r]
            for c in range(16):
                k_c = c // 4
                i_c = c % 4
                W_new[(4 + k_c) * 4 + i_c][56 + r] = W2[c][r]

        # Block 3: col 8 of B (4x1 sub-problem), rank 16 (schoolbook)
        for i in range(4):
            for j in range(4):
                r = 112 + i * 4 + j
                U_new[i * 4 + j][r] = 1
                V_new[j * 9 + 8][r] = 1
                W_new[8 * 4 + i][r] = 1

        if verify_numpy(U_new, V_new, W_new, n, m, p):
            best_rank = 128
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    # Try to find a better decomposition using remaining time
    # Strategy: Try different splits of the 9 columns
    # 9 = 2+2+2+2+1: 4*(4x4x2) + 4x4x1
    # 4x4x2: use 4 Strassen 2x2x2 blocks = 28
    # Total: 4*28 + 16 = 128. Same.
    
    # Try: 9 = 4+3+2: 4x4x4(56) + 4x4x3(48) + 4x4x2(32) = 136. Worse.
    # Try: 9 = 4+5: 56 + 72 = 128. Same.
    
    # The best we can do with Strassen 2x2x2 is 128.
    # To go below 128, we need a better 4x4x4 decomposition (rank < 56)
    # or a different approach entirely.
    
    # For now, return the rank-128 solution.
    if time.time() < deadline - 5:
        try:
            if not verify_numpy(best_result["U"], best_result["V"], best_result["W"], n, m, p):
                best_rank = rank_school
                best_result = {"U": U, "V": V, "W": W}
        except:
            best_rank = rank_school
            best_result = {"U": U, "V": V, "W": W}

    return best_result

# EVOLVE-BLOCK-END
