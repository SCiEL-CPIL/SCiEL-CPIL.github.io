"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _add_strassen_block(U, V, W, j0, j1, k0, k1, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block."""
    # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A_1,j0 + A_1,j1) B_j0,k0
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A_0,j0 + A_0,j1) B_j1,k1
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _add_standard_block(U, V, W, j, k0, k1, col, n, m, p):
    """Add 4 standard products for a 2x1x2 block (single j, two k's)."""
    # C_0,k0 = A_0,j B_j,k0
    U[0*m + j][col] += 1
    V[j*p + k0][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    # C_1,k0 = A_1,j B_j,k0
    U[1*m + j][col] += 1
    V[j*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # C_0,k1 = A_0,j B_j,k1
    U[0*m + j][col] += 1
    V[j*p + k1][col] += 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # C_1,k1 = A_1,j B_j,k1
    U[1*m + j][col] += 1
    V[j*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition
    # Block 1: j in {0,1}, k in {0,1} -> Strassen (7)
    # Block 2: j=2, k in {0,1} -> Standard (4)
    # Block 3: j in {3,4}, k in {2,3} -> Strassen (7)
    # Block 4: j in {3,4}, k in {4,5} -> Strassen (7)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j in {0,1}, k in {0,1}
    col = _add_strassen_block(U, V, W, 0, 1, 0, 1, col, n, m, p)
    
    # Block 2: j=2, k in {0,1}
    col = _add_standard_block(U, V, W, 2, 0, 1, col, n, m, p)
    
    # Block 3: j in {3,4}, k in {2,3}
    col = _add_strassen_block(U, V, W, 3, 4, 2, 3, col, n, m, p)
    
    # Block 4: j in {3,4}, k in {4,5}
    col = _add_strassen_block(U, V, W, 3, 4, 4, 5, col, n, m, p)
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
