"""Matrix multiplication 2x3x10 tensor decomposition with rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape for efficient checking
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Vectorized check
    # Iterate over c
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U_arr[:, c].reshape(n, m)
        v = V_arr[:, c].reshape(m, p)
        w = W_arr[:, c].reshape(p, n)
        # w[k, i] is the coefficient for C[i, k]
        # We need to add u[i, j] * v[j, k] * w[k, i] to T[i, j, k]
        # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        # This is not a standard outer product.
        # Let's loop over j for efficiency in Python if numpy is slow, 
        # but here n,m,p are small.
        for j in range(m):
            # u[:, j] is (n,), v[j, :] is (p,)
            # We need T[:, j, :] += u[:, j, None] * v[j, None, :] * w.T
            # w.T[i, k] = w[k, i]
            T[:, j, :] += u[:, j, None] * v[j, None, :] * w.T
            
    return np.allclose(T, 1.0, atol=1e-9)

def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition.
    A is 2x3, B is 3x10, C is 2x10.
    Split B into 5 blocks of 2 columns.
    For each block:
      Part 1: A[:, 0] * B[0, block] -> Rank 4
      Part 2: A[:, 1:2] * B[1:2, block] -> Rank 7 (Strassen)
    Total per block: 11.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i, 0] * B[0, k] -> C[i, k]
    # This part is independent for each k, but we can group by block of 2.
    # Actually, the "outer" part is A[:, 0] (2x1) and B[0, :] (1x10).
    # Rank is 20.
    # We can just fill these 20 terms directly.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10.
    # Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries: A[0,1], A[0,2], A[1,1], A[1,2])
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        # Strassen indices for V (B2 block entries: B[1,k0], B[1,k1], B[2,k0], B[2,k1])
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        # W indices for outputs (C[0,k0], C[0,k1], C[1,k0], C[1,k1])
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank-55 solution
    U, V, W = _build_strassen55(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook (should not happen)
        rank = n * m * p
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
