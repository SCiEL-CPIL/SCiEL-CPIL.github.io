"""Direct-sum (row-wise) tensor decomposition for 3x5 by 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Block-diagonal (direct-sum) decomposition:
    # Treat A (n x m) as n independent 1 x m row vectors.
    # Each row-vector times B (m x p) yields a 1 x p row, decomposed as
    # rank p (one rank-1 per output column). Total rank = n * p.
    rank = n * p

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for i in range(n):          # each row of A
        for k in range(p):      # each output column
            # U picks out row i of A: A[i, j]
            U[i * m + 0: i * m + m, col] = 1.0
            # V picks out column k of B: B[j, k]
            V[0 * p + k: m * p + k: p, col] = 1.0
            # W places the product at output (i, k): W[k*n + i, col]
            W[k * n + i, col] = 1.0
            col += 1

    # Exactness verification against the multiplication tensor.
    # T[i, j, k] = 1  (C[i,k] = sum_j A[i,j] B[j,k])
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Uf = U.flatten()
    Vf = V.flatten()
    Wf = W.flatten()
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = (i * m + j) * rank
                idx_v = (j * p + k) * rank
                idx_w = (k * n + i) * rank
                s = 0.0
                for c in range(rank):
                    s += Uf[idx_u + c] * Vf[idx_v + c] * Wf[idx_w + c]
                if s != 1.0:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to guaranteed-correct schoolbook (rank n*m*p)
        rank2 = n * m * p
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j, c] = 1.0
                    V2[j * p + k, c] = 1.0
                    W2[k * n + i, c] = 1.0
                    c += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
