"""Matrix multiplication 2,2,11 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify():
    n, m, p = DIMENSIONS
    d = construct()
    U = np.array(d["U"], dtype=float)
    V = np.array(d["V"], dtype=float)
    W = np.array(d["W"], dtype=float)
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = float(np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]))
                assert abs(val - 1.0) < 1e-9, f"mismatch at {i,j,k}: {val}"
    # Check off-diagonal entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # determine if (a,b,c) corresponds to a valid (i,j,k)
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j == j2 and k == k2 and i == i2:
                    continue  # diagonal, already checked
                val = float(np.sum(U[a, :] * V[b, :] * W[c, :]))
                assert abs(val) < 1e-9, f"off-diag nonzero at {a,b,c}: {val}"
    return rank

# EVOLVE-BLOCK-END
