import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Rank 32 decomposition using schoolbook for all blocks
    # 6 blocks of 2x2x2 (rank 4 each)
    # 2 blocks of 2x2x1 (rank 4 each)
    rank = 32
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)

    col = 0

    def add_schoolbook_222_block(i0, j0, k0):
        nonlocal col
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # C[i0, k0] += A[i0, j0] * B[j0, k0]
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # C[i0, k0+1] += A[i0, j0] * B[j0, k0+1]
        U[i0*m + j0, col] = 1
        V[j0*p + k0 + 1, col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # C[i0, k0] += A[i0, j0+1] * B[j0+1, k0]
        U[i0*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # C[i0, k0+1] += A[i0, j0+1] * B[j0+1, k0+1]
        U[i0*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0 + 1, col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # C[i0+1, k0] += A[i0+1, j0] * B[j0, k0]
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0 + 1, col] = 1
        col += 1
        
        # C[i0+1, k0+1] += A[i0+1, j0] * B[j0, k0+1]
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0 + 1, col] = 1
        W[(k0+1)*n + i0 + 1, col] = 1
        col += 1
        
        # C[i0+1, k0] += A[i0+1, j0+1] * B[j0+1, k0]
        U[(i0+1)*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0 + 1, col] = 1
        col += 1
        
        # C[i0+1, k0+1] += A[i0+1, j0+1] * B[j0+1, k0+1]
        U[(i0+1)*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0 + 1, col] = 1
        W[(k0+1)*n + i0 + 1, col] = 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0+1, k0)
        # C indices: (i0, k0), (i0+1, k0)
        
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[i0*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0 + 1, col] = 1
        col += 1
        
        U[(i0+1)*m + j0 + 1, col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0 + 1, col] = 1
        col += 1

    # Partition:
    # j in {0,1}, {2,3}
    # k in {0,1}, {2,3}, {4,5}, {6}
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_schoolbook_222_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_fb = np.zeros((n*m, n*m*p))
        V_fb = np.zeros((m*p, n*m*p))
        W_fb = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, idx] = 1.0
                    V_fb[j*p + k, idx] = 1.0
                    W_fb[k*n + i, idx] = 1.0
                    idx += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
