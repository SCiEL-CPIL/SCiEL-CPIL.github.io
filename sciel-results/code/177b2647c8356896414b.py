import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    rank = 42
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    col = 0
    
    # Strassen 2x2x2: given A block at (i0, j0) and B block at (j0, k0),
    # produce 7 products and map them to C output.
    # A = [[A11, A12], [A21, A22]] where A[i,j] is at U index i*m + j
    # B = [[B11, B12], [B21, B22]] where B[j,k] is at V index j*p + k
    # C = [[C11, C12], [C21, C22]] where C[i,k] is at W index k*n + i
    #
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_block(i0, j0, k0, start_col):
        """
        i0: top row of A block (0 or 1)
        j0: top col of A block / top row of B block (0, 2)
        k0: top col of B block (0, 2, 4)
        start_col: starting column index in U, V, W
        Returns the next column index.
        """
        # A indices: A[i0, j0], A[i0, j0+1], A[i0+1, j0], A[i0+1, j0+1]
        # B indices: B[j0, k0], B[j0, k0+1], B[j0+1, k0], B[j0+1, k0+1]
        # C indices: C[i0, k0], C[i0, k0+1], C[i0+1, k0], C[i0+1, k0+1]
        
        a11 = i0 * m + j0
        a12 = i0 * m + j0 + 1
        a21 = (i0 + 1) * m + j0
        a22 = (i0 + 1) * m + j0 + 1
        
        b11 = j0 * p + k0
        b12 = j0 * p + k0 + 1
        b21 = (j0 + 1) * p + k0
        b22 = (j0 + 1) * p + k0 + 1
        
        c11 = k0 * n + i0
        c12 = k0 * n + i0 + 1  # Wait, C[i,k] is at W index k*n + i
        # C[i0, k0] -> W[k0*n + i0]
        # C[i0, k0+1] -> W[(k0+1)*n + i0]
        # C[i0+1, k0] -> W[k0*n + i0+1]
        # C[i0+1, k0+1] -> W[(k0+1)*n + i0+1]
        c11 = k0 * n + i0
        c12 = (k0 + 1) * n + i0
        c21 = k0 * n + i0 + 1
        c22 = (k0 + 1) * n + i0 + 1
        
        # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        c = start_col
        U[a11][c] += 1; V[b11][c] += 1; W[c11][c] += 1; W[c22][c] += 1
        c += 1
        U[a11][c] += 1; V[b22][c] += 1; W[c11][c] += 1; W[c22][c] += 1
        c += 1
        U[a22][c] += 1; V[b11][c] += 1; W[c11][c] += 1; W[c22][c] += 1
        c += 1
        U[a22][c] += 1; V[b22][c] += 1; W[c11][c] += 1; W[c22][c] += 1
        c += 1
        
        # M2 = (A21+A22)B11 = A21*B11 + A22*B11
        U[a21][c] += 1; V[b11][c] += 1; W[c21][c] += 1; W[c22][c] += -1
        c += 1
        U[a22][c] += 1; V[b11][c] += 1; W[c21][c] += 1; W[c22][c] += -1
        c += 1
        
        # M3 = A11(B12-B22) = A11*B12 - A11*B22
        U[a11][c] += 1; V[b12][c] += 1; W[c12][c] += 1; W[c22][c] += 1
        c += 1
        U[a11][c] += 1; V[b22][c] += -1; W[c12][c] += 1; W[c22][c] += 1
        c += 1
        
        # M4 = A22(B21-B11) = A22*B21 - A22*B11
        U[a22][c] += 1; V[b21][c] += 1; W[c11][c] += 1; W[c21][c] += 1
        c += 1
        U[a22][c] += 1; V[b11][c] += -1; W[c11][c] += 1; W[c21][c] += 1
        c += 1
        
        # M5 = (A11+A12)B22 = A11*B22 + A12*B22
        U[a11][c] += 1; V[b22][c] += 1; W[c12][c] += 1; W[c11][c] += -1
        c += 1
        U[a12][c] += 1; V[b22][c] += 1; W[c12][c] += 1; W[c11][c] += -1
        c += 1
        
        # M6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        U[a21][c] += 1; V[b11][c] += 1; W[c22][c] += 1
        c += 1
        U[a21][c] += 1; V[b12][c] += 1; W[c22][c] += 1
        c += 1
        U[a11][c] += -1; V[b11][c] += 1; W[c22][c] += 1
        c += 1
        U[a11][c] += -1; V[b12][c] += 1; W[c22][c] += 1
        c += 1
        
        # M7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        U[a12][c] += 1; V[b21][c] += 1; W[c11][c] += 1
        c += 1
        U[a12][c] += 1; V[b22][c] += 1; W[c11][c] += 1
        c += 1
        U[a22][c] += -1; V[b21][c] += 1; W[c11][c] += 1
        c += 1
        U[a22][c] += -1; V[b22][c] += 1; W[c11][c] += 1
        c += 1
        
        return c
    
    # We have 6 blocks: 2 (j-groups) × 3 (k-groups)
    # j-group 0: j0=0 (A cols 0,1; B rows 0,1)
    # j-group 1: j0=2 (A cols 2,3; B rows 2,3)
    # k-groups: k0=0, 2, 4
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            for i0 in [0]:  # Only one i-group since n=2
                col = strassen_block(i0, j0, k0, col)
    
    # Verify: col should be 42
    assert col == 42, f"Expected 42 columns, got {col}"
    
    # Verify correctness
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1) > 1e-10:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}, expected 1")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
