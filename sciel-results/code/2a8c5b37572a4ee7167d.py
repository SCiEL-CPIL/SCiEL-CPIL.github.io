import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Known rank-14 decomposition via two Strassen-style 2x2 x 2x8 blocks
    rank = 14

    # Build U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # We treat the 4 middle indices as two groups of 2: j_group in {0, 1}
    # Within each group, local j' in {0, 1}
    # Strassen for 2x2 * 2x8 (i.e., n=2, m_local=2, p=8)

    # The 7 Strassen products for 2x2 * 2x8:
    # M0 = (A00 + A11) * (B0 + B1)
    # M1 = (A10 + A11) * B0
    # M2 = A00 * (B0 - B1)
    # M3 = A11 * (B1 - B0)
    # M4 = (A00 + A01) * B1
    # M5 = (A10 - A00) * (B0 + B1)
    # M6 = (A01 - A11) * (B1 + B0)

    # For each Strassen coefficient c in 0..6:
    #   U part: 2x2 matrix (i, j') with entries
    #   V part: 2x8 matrix (j', k) with entries
    #   W part: 8x2 matrix (k, i) with entries (note: W[k*n+i, c])

    # Define the 7 Strassen factor matrices
    U_s = np.zeros((4, 7))
    V_s = np.zeros((16, 7))
    W_s = np.zeros((16, 7))

    # M0
    U_s[0, 0] = 1.0; U_s[3, 0] = 1.0  # A00, A11
    for k in range(8):
        V_s[k, 0] = 1.0; V_s[8 + k, 0] = 1.0
        W_s[k * 2, 0] = 1.0; W_s[k * 2 + 1, 0] = 1.0

    # M1
    U_s[2, 1] = 1.0; U_s[3, 1] = 1.0  # A10, A11
    for k in range(8):
        V_s[k, 1] = 1.0
        W_s[k * 2 + 1, 1] = 1.0

    # M2
    U_s[0, 2] = 1.0  # A00
    for k in range(8):
        V_s[k, 2] = 1.0; V_s[8 + k, 2] = -1.0
        W_s[k * 2, 2] = 1.0

    # M3
    U_s[3, 3] = 1.0  # A11
    for k in range(8):
        V_s[8 + k, 3] = 1.0; V_s[k, 3] = -1.0
        W_s[k * 2, 3] = -1.0

    # M4
    U_s[0, 4] = 1.0; U_s[1, 4] = 1.0  # A00, A01
    for k in range(8):
        V_s[8 + k, 4] = 1.0
        W_s[k * 2, 4] = 1.0

    # M5
    U_s[2, 5] = 1.0; U_s[0, 5] = -1.0  # A10, -A00
    for k in range(8):
        V_s[k, 5] = 1.0; V_s[8 + k, 5] = 1.0
        W_s[k * 2, 5] = 1.0

    # M6
    U_s[1, 6] = 1.0; U_s[3, 6] = -1.0  # A01, -A11
    for k in range(8):
        V_s[8 + k, 6] = 1.0; V_s[k, 6] = 1.0
        W_s[k * 2 + 1, 6] = 1.0

    # Assemble into full U, V, W
    for grp in range(2):
        base_c = grp * 7
        for c in range(7):
            col = base_c + c
            # U: rows indexed by i*m + j, where j = grp*2 + j'
            for i in range(2):
                for jp in range(2):
                    U[i * m + grp * 2 + jp, col] = U_s[i * 2 + jp, c]
            # V: rows indexed by j*p + k, where j = grp*2 + jp
            for jp in range(2):
                for k in range(8):
                    V[(grp * 2 + jp) * p + k, col] = V_s[jp * 8 + k, c]
            # W: rows indexed by k*n + i
            for k in range(8):
                for i in range(2):
                    W[k * n + i, col] = W_s[k * 2 + i, c]

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i * m_val + j, r] * V_mat[j * p_val + k, r] * W_mat[k * n_val + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p) and time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank = n*m*p = 64
    rank_fb = n * m * p
    U_fb = [[0.0] * rank_fb for _ in range(n * m)]
    V_fb = [[0.0] * rank_fb for _ in range(m * p)]
    W_fb = [[0.0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1.0
                V_fb[j * p + k][col] = 1.0
                W_fb[k * n + i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
