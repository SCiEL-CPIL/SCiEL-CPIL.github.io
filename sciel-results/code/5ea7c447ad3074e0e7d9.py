import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Rank calculation:
    # M_2,2,2 (Strassen) = 7
    # M_2,1,2 (Outer) = 2*1*2 = 4
    # M_2,2,2 (Strassen) = 7
    # M_2,2,1 (Outer) = 2*2*1 = 4
    # M_2,1,3 (Outer) = 2*1*3 = 6
    # Total = 28
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # Strassen formulas for 2x2 * 2x2
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_outer_product(i_indices, j_idx, k_indices):
        nonlocal col
        for i in i_indices:
            for k in k_indices:
                U[i*m + j_idx, col] = 1
                V[j_idx*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    # 1. A[:, 0:2] * B[0:2, 0:2] -> Strassen
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # 2. A[:, 2] * B[2, 0:2] -> Outer
    add_outer_product([0, 1], 2, [0, 1])
    
    # 3. A[:, 0:2] * B[0:2, 2:4] -> Strassen
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # 4. A[:, 0:2] * B[0:2, 4] -> Outer
    add_outer_product([0, 1], 0, [4])
    add_outer_product([0, 1], 1, [4])
    
    # 5. A[:, 2] * B[2, 2:5] -> Outer
    add_outer_product([0, 1], 2, [2, 3, 4])

    # Verification
    res = np.zeros((n, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += np.einsum('ij,jk,ki->ik', U_c, V_c, W_c)
        
    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
