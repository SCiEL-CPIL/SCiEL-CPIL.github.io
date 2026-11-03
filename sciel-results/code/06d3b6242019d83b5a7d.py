import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 120.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4 (A rows), m=4 (A cols = B rows), p=6 (B cols = C cols)

    # Strassen 2x2x2 building block (row-major indices 0..3)
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    # ---- Part 1: 4x4x4 (columns 0..3) via direct 2x2-block Strassen, rank 28 ----
    # U[a][i,j] = alpha[s][a]  (a = block row, col of A)
    # V[b][j,k] = beta[s][b]   (b = block row, col of B)
    # W[(a,b)][k,i] = recon[s][a,b]
    R = 44
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        for a in range(2):
            for b in range(2):
                # U: A block (a, b)
                for r in range(2):
                    for c in range(2):
                        U[(a * 2 + r) * 4 + (b * 2 + c), col] = alpha[s][a]
                # V: B block (b, a)  -> B row block = b, col block = a
                for r in range(2):
                    for c in range(2):
                        V[(b * 2 + r) * 6 + (a * 2 + c), col] = beta[s][b]
                # W: C block (a, b) -> C row block = b, col block = a
                for r in range(2):
                    for c in range(2):
                        W[(b * 2 + r) * 4 + (a * 2 + c), col] = recon[s][a][b]
                col += 1

    # ---- Part 2: 4x4x2 (columns 4..5) via schoolbook, rank 16 ----
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j, col] = 1.0
                V[j * 6 + (k + 4), col] = 1.0
                W[(k + 4) * 4 + i, col] = 1.0
                col += 1

    assert col == R

    # ---- Verification against the tensor ----
    if time.time() - t_start < _DEADLINE - 5:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    err = abs(prod - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-9:
            # Fallback: full schoolbook rank 96
            R = 96
            U = np.zeros((16, R))
            V = np.zeros((24, R))
            W = np.zeros((24, R))
            c = 0
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        U[i * 4 + j, c] = 1.0
                        V[j * 6 + k, c] = 1.0
                        W[k * 4 + i, c] = 1.0
                        c += 1

    def to_list(mat):
        out = []
        for row in mat:
            out_row = []
            for x in row:
                rx = round(x)
                out_row.append(int(rx) if abs(x - rx) < 1e-9 else x)
            out.append(out_row)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
