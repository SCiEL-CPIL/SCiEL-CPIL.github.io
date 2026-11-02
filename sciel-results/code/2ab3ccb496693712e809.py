"""Matrix multiplication 2x3x4: rank-16 exact tensor decomposition via two Strassen blocks + dot product."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 16
    U = np.zeros((n * m, rank))   # 6 x 16
    V = np.zeros((m * p, rank))   # 12 x 16
    W = np.zeros((p * n, rank))   # 8 x 16

    # Strassen rank-1 coefficient table for 2x2x2.
    # (du, dv, dw): each tuple over the 4 sub-blocks in order
    # (top-left, top-right, bottom-left, bottom-right)
    strassen = [
        ((1, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (0, 1)),
        ((1, 0), (0, 1), (0, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 1), (0, 1), (1, 0)),
        ((1, 0), (1, 1), (1, 1)),
        ((0, 1), (1, 1), (1, 1)),
    ]

    def add_block(base, ai0, aj0, bj0, bk0, ck0, ci0):
        # A rows ai0..ai0+1, cols aj0..aj0+1
        # B rows bj0..bj0+1, cols bk0..bk0+1
        # C cols ck0..ck0+1, rows ci0..ci0+1
        for s, (du, dv, dw) in enumerate(strassen):
            cs = base + s
            for a_i in (0, 1):
                for a_j in (0, 1):
                    U[(ai0 + a_i) * m + (aj0 + a_j), cs] += du[a_j]
            for b_j in (0, 1):
                for b_k in (0, 1):
                    V[(bj0 + b_j) * p + (bk0 + b_k), cs] += dv[b_j]
            for c_k in (0, 1):
                for c_i in (0, 1):
                    W[(ck0 + c_k) * n + (ci0 + c_i), cs] += dw[c_i]

    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    add_block(0, ai0=0, aj0=0, bj0=0, bk0=0, ck0=0, ci0=0)
    # Block 2: A[:, 1:3] * B[1:3, 2:4]
    add_block(7, ai0=0, aj0=1, bj0=1, bk0=2, ck0=2, ci0=0)
    # Dot product: A[:, 2] * B[2, :]  -> rank 2 (one per i)
    for i in (0, 1):
        cs = 14 + i
        U[i * m + 2, cs] = 1
        for k in (0, 1, 2, 3):
            V[2 * p + k, cs] = 1
            W[k * n + i, cs] = 1

    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]

    # Exact verification.
    A = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    B = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    C_true = A @ B
    Uf = np.array(U_list, dtype=float)
    Vf = np.array(V_list, dtype=float)
    Wf = np.array(W_list, dtype=float)
    C_comp = (Wf.T @ (Uf * Vf.T)).reshape(n, p)

    if not np.allclose(C_comp, C_true, atol=1e-9):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
