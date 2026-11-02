import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Rank-56 block-Strassen: 8 independent 2x2 Strassen blocks.
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)

    # Define Strassen factors for 2x2 matrices
    # U, V, W are 4x7 matrices for the 2x2 case
    # Indices: U: (i,j) -> 0..3, V: (j,k) -> 0..3, W: (k,i) -> 0..3
    s_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 1, 0, 0, 1, 0, 1],   # 01
        [1, 1, 0, 1, 0, 1, 0],   # 10
        [0, 0, 1, 1, 0, 0, -1]   # 11
    ], dtype=float)
    
    s_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 1, 0, 0, 1],   # 01
        [1, 0, 1, 0, 1, 1, 0],   # 10
        [0, 0, 0, 1, 0, 0, 1]    # 11
    ], dtype=float)
    
    s_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # 00
        [0, 1, 1, 0, 1, 0, 0],   # 01
        [1, -1, 0, 1, 0, 1, 0],  # 10
        [0, 0, 1, 0, 0, 0, 1]    # 11
    ], dtype=float)

    # Map 2x2 indices to global indices
    # U global index: i*m + j. i in 0,1. j in 0..3.
    # V global index: j*p + k. j in 0..3. k in 0..7.
    # W global index: k*n + i. k in 0..7. i in 0,1.

    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            
            # Local indices for U (i,j)
            # j corresponds to 2*a and 2*a+1
            u_j0 = 2 * a
            u_j1 = 2 * a + 1
            u_idx = np.array([
                0 * m + u_j0, 0 * m + u_j1,
                1 * m + u_j0, 1 * m + u_j1
            ])
            
            # Local indices for V (j,k)
            # j corresponds to 2*a and 2*a+1
            # k corresponds to 2*b and 2*b+1
            v_j0 = 2 * a
            v_j1 = 2 * a + 1
            v_k0 = 2 * b
            v_k1 = 2 * b + 1
            v_idx = np.array([
                v_j0 * p + v_k0, v_j0 * p + v_k1,
                v_j1 * p + v_k0, v_j1 * p + v_k1
            ])
            
            # Local indices for W (k,i)
            # k corresponds to 2*b and 2*b+1
            # i corresponds to 0 and 1
            w_k0 = 2 * b
            w_k1 = 2 * b + 1
            w_idx = np.array([
                w_k0 * n + 0, w_k1 * n + 0,
                w_k0 * n + 1, w_k1 * n + 1
            ])

            # Fill the matrices
            U0[u_idx, base:base+7] = s_U
            V0[v_idx, base:base+7] = s_V
            W0[w_idx, base:base+7] = s_W

    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U0, V0, W0):
        # Fallback
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
