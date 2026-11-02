import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Build rank-56 Strassen-based decomposition
    rank_56 = 56
    U_56 = np.zeros((nm, rank_56))
    V_56 = np.zeros((mp, rank_56))
    W_56 = np.zeros((pn, rank_56))

    # Strassen terms for 2x2 * 2x2
    U_s = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ])
    V_s = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 1, 0, -1],  # M3: b01-b11
        [-1, 0, 1, 0],  # M4: b10-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1]    # M7: b10+b11
    ])
    W_s = np.array([
        [1, 0, 0, 1],   # M1: C00+, C11+
        [0, 0, 1, -1],  # M2: C10+, C11-
        [0, 1, 0, 1],   # M3: C01+, C11+
        [-1, 1, 0, 0],  # M4: C00-, C01+
        [1, 0, 1, 0],   # M5: C00+, C10+
        [0, 0, 0, 0],   # M6: not used
        [-1, 0, 0, 1]   # M7: C00-, C11+
    ])

    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for s in range(7):
                U_56[0 * m + a_block * 2 + 0, col] = U_s[s, 0]
                U_56[0 * m + a_block * 2 + 1, col] = U_s[s, 1]
                U_56[1 * m + a_block * 2 + 0, col] = U_s[s, 2]
                U_56[1 * m + a_block * 2 + 1, col] = U_s[s, 3]

                V_56[(a_block * 2 + 0) * p + b_block * 2 + 0, col] = V_s[s, 0]
                V_56[(a_block * 2 + 0) * p + b_block * 2 + 1, col] = V_s[s, 1]
                V_56[(a_block * 2 + 1) * p + b_block * 2 + 0, col] = V_s[s, 2]
                V_56[(a_block * 2 + 1) * p + b_block * 2 + 1, col] = V_s[s, 3]

                W_56[(b_block * 2 + 0) * n + 0, col] = W_s[s, 0]
                W_56[(b_block * 2 + 0) * n + 1, col] = W_s[s, 1]
                W_56[(b_block * 2 + 1) * n + 0, col] = W_s[s, 2]
                W_56[(b_block * 2 + 1) * n + 1, col] = W_s[s, 3]

                col += 1

    # Verify
    U_t = U_56.T.reshape(n, m, rank_56)
    V_t = V_56.T.reshape(m, p, rank_56)
    W_t = W_56.T.reshape(p, n, rank_56)
    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    target = np.ones((n, m, p))

    if np.allclose(result, target, atol=1e-10):
        return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
    else:
        # Fallback: schoolbook rank 64
        rank_school = 64
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
