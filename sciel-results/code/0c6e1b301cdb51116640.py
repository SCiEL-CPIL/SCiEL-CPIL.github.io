import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # 4 disjoint Strassen 2x2x2 blocks
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 2: i=0,1; j=2,3; k=0,1
    # Block 3: i=0,1; j=0,1; k=2,3
    # Block 4: i=0,1; j=2,3; k=2,3
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]

    # Calculate number of uncovered entries for schoolbook
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Total entries: 3 * 5 * 5 = 75
    # Covered entries: 2 * 4 * 4 = 32
    # Uncovered entries: 75 - 32 = 43
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    schoolbook_rank = n * m * p - (2 * 4 * 4)
    total_rank = strassen_rank + schoolbook_rank

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Add schoolbook terms for uncovered entries
    # Uncovered if NOT (i < 2 AND j < 4 AND k < 4)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not (i < 2 and j < 4 and k < 4):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1

    # Verification (optional but good for debugging locally, keep for safety if time permits)
    if time.time() - start < 10:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            target = np.zeros((n*m, m*p, p*n), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = 1
            T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
            if not np.allclose(T_decomp, target, atol=1e-10):
                # If verification fails, fall back to pure schoolbook (rank 75)
                rank = n * m * p
                U_f = [[0]*rank for _ in range(n*m)]
                V_f = [[0]*rank for _ in range(m*p)]
                W_f = [[0]*rank for _ in range(p*n)]
                c = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_f[i*m+j][c] = 1
                            V_f[j*p+k][c] = 1
                            W_f[k*n+i][c] = 1
                            c += 1
                return {"U": U_f, "V": V_f, "W": W_f}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
