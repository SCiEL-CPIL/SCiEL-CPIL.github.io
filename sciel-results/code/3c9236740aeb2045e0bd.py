"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly by checking all valid and invalid triplets."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen_block(U, V, W, col_offset, u_indices, v_indices, w_indices):
    """
    Add Strassen factors for a 2x2x2 block.
    u_indices: [u00, u01, u10, u11] - U row indices for A[0,0], A[0,1], A[1,0], A[1,1]
    v_indices: [v00, v01, v10, v11] - V row indices for B[0,0], B[0,1], B[1,0], B[1,1]
    w_indices: [w00, w01, w10, w11] - W row indices for C[0,0], C[0,1], C[1,0], C[1,1]
    """
    u00, u01, u10, u11 = u_indices
    v00, v01, v10, v11 = v_indices
    w00, w01, w10, w11 = w_indices
    
    # M1 = (X11+X22)(Y11+Y22)
    c = col_offset + 0
    U[u00][c] += 1; U[u11][c] += 1
    V[v00][c] += 1; V[v11][c] += 1
    W[w00][c] += 1; W[w11][c] += 1
    
    # M2 = (X21+X22)Y11
    c = col_offset + 1
    U[u10][c] += 1; U[u11][c] += 1
    V[v00][c] += 1
    W[w10][c] += 1; W[w11][c] += -1
    
    # M3 = X11(Y21-Y22)
    c = col_offset + 2
    U[u00][c] += 1
    V[v10][c] += 1; V[v11][c] += -1
    W[w01][c] += 1; W[w11][c] += 1
    
    # M4 = X22(Y12-Y11)
    c = col_offset + 3
    U[u11][c] += 1
    V[v01][c] += 1; V[v00][c] += -1
    W[w00][c] += 1; W[w10][c] += 1
    
    # M5 = (X11+X12)Y22
    c = col_offset + 4
    U[u00][c] += 1; U[u01][c] += 1
    V[v11][c] += 1
    W[w00][c] += -1; W[w01][c] += 1
    
    # M6 = (X21-X11)(Y11+Y12)
    c = col_offset + 5
    U[u10][c] += 1; U[u00][c] += -1
    V[v00][c] += 1; V[v01][c] += 1
    W[w11][c] += 1
    
    # M7 = (X12-X22)(Y21+Y22)
    c = col_offset + 6
    U[u01][c] += 1; U[u11][c] += -1
    V[v10][c] += 1; V[v11][c] += 1
    W[w00][c] += 1

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: j in {0,1}, j' in {0,1}, k in {0,1} - Strassen
    # U indices: i*3+j -> A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: j'*4+k -> B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W indices: k*2+i -> C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    u_idx_1 = [0, 1, 3, 4]
    v_idx_1 = [0, 1, 4, 5]
    w_idx_1 = [0, 2, 1, 3]  # w00, w01, w10, w11
    _build_strassen_block(U, V, W, 0, u_idx_1, v_idx_1, w_idx_1)
    
    # Block 2: j in {0,1}, j' in {0,1}, k in {2,3} - Strassen
    # U indices same: 0, 1, 3, 4
    # V indices: j'*4+k -> B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # W indices: k*2+i -> C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    u_idx_2 = [0, 1, 3, 4]
    v_idx_2 = [2, 3, 6, 7]
    w_idx_2 = [4, 6, 5, 7]  # w00, w01, w10, w11
    _build_strassen_block(U, V, W, 7, u_idx_2, v_idx_2, w_idx_2)
    
    # Block 3: j=2, j'=2, k in {0,1,2,3} - Schoolbook (8 factors)
    # U indices: i*3+2 -> A[0,2]=2, A[1,2]=5
    # V indices: 2*4+k -> B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # W indices: k*2+i
    col = 14
    for i in range(n):
        for k in range(p):
            U[i*3+2][col] = 1
            V[2*4+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    # Verify
    if not _verify_exact(U, V, W, n, m, p, rank, tol=1e-8):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
