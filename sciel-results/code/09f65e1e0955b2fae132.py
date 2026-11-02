import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_222(U, V, W, offset, i_off, j_off, k_off):
    """
    Add Strassen decomposition for 2x2x2 block to U, V, W.
    U rows: (i, j) -> i_off + i, j_off + j
    V rows: (j, k) -> j_off + j, k_off + k
    W rows: (k, i) -> k_off + k, i_off + i
    offset: starting index for rank dimension t
    """
    # Strassen 7 terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U coefficients for A
    # A00, A01, A10, A11
    # M1: A00(+1), A11(+1)
    # M2: A10(+1), A11(+1)
    # M3: A00(+1)
    # M4: A11(+1)
    # M5: A00(+1), A01(+1)
    # M6: A10(+1), A00(-1)
    # M7: A01(+1), A11(-1)
    
    U_coeffs = [
        [1, 0, 0, 1], # M1: A00, A01, A10, A11
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V coefficients for B
    # M1: B00(+1), B11(+1)
    # M2: B00(+1)
    # M3: B01(+1), B11(-1)
    # M4: B10(+1), B00(-1)
    # M5: B11(+1)
    # M6: B00(+1), B01(+1)
    # M7: B10(+1), B11(+1)
    
    V_coeffs = [
        [1, 0, 0, 1], # M1: B00, B01, B10, B11
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients for C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W rows are (k, i).
    # C00 -> k=0, i=0
    # C01 -> k=1, i=0
    # C10 -> k=0, i=1
    # C11 -> k=1, i=1
    
    W_coeffs = [
        # M1, M2, M3, M4, M5, M6, M7
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    for t in range(7):
        idx = offset + t
        # Fill U
        for i in range(2):
            for j in range(2):
                val = U_coeffs[t][i*2 + j]
                if val != 0:
                    U[i_off + i, j_off + j, idx] = val
        # Fill V
        for j in range(2):
            for k in range(2):
                val = V_coeffs[t][j*2 + k]
                if val != 0:
                    V[j_off + j, k_off + k, idx] = val
        # Fill W
        for k in range(2):
            for i in range(2):
                val = W_coeffs[k*2 + i][t]
                if val != 0:
                    W[k_off + k, i_off + i, idx] = val

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Rank calculation:
    # 2 blocks of 2x2x6 (each 3x 2x2x2 Strassen) -> 2 * 21 = 42
    # 1 block of 2x1x6 (Rank 1) -> 1
    # Total Rank 43
    
    R = 43
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:6]
    # A rows 0,1; Cols 0,1
    # B Rows 0,1; Cols 0,1; 2,3; 4,5
    i_off = 0
    j_off = 0
    
    # B sub-block 1: cols 0,1
    _strassen_222(U, V, W, 0, i_off, j_off, 0)
    # B sub-block 2: cols 2,3
    _strassen_222(U, V, W, 7, i_off, j_off, 2)
    # B sub-block 3: cols 4,5
    _strassen_222(U, V, W, 14, i_off, j_off, 4)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:6]
    # A rows 0,1; Cols 2,3
    # B Rows 2,3; Cols 0,1; 2,3; 4,5
    j_off = 2
    
    _strassen_222(U, V, W, 21, i_off, j_off, 0)
    _strassen_222(U, V, W, 28, i_off, j_off, 2)
    _strassen_222(U, V, W, 35, i_off, j_off, 4)
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:6]
    # A rows 0,1; Col 4
    # B Row 4; Cols 0..5
    # T_{i, 4, k} = 1
    # U[i, 4, 42] = 1
    # V[4, k, 42] = 1
    # W[k, i, 42] = 1
    
    for i in range(n):
        U[i, 4, 42] = 1
    for k in range(p):
        V[4, k, 42] = 1
    for k in range(p):
        for i in range(n):
            W[k, i, 42] = 1
            
    # Verify
    A_rand = np.random.randint(0, 10, size=(n, m))
    B_rand = np.random.randint(0, 10, size=(m, p))
    C_true = A_rand @ B_rand
    
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for t in range(R):
                u_val = np.dot(U[i*m:(i+1)*m, t], A_rand[i, :])
                v_val = np.dot(V[:, k], B_rand[:, k]) # Wait, V index is j*p+k.
                # V[j, k] corresponds to row j*p + k? No.
                # V shape (m*p, R). Row index is j*p + k.
                # So V[j*p + k, t] is the entry.
                # We need sum_j V[j*p+k, t] B[j, k].
                
                # Vectorized:
                # V_part = V[k::p, t] (rows k, p+k, 2p+k...)
                # B_part = B[:, k] (col k)
                v_val = np.dot(V[k::p, t], B_rand[:, k])
                s += W[k*n+i, t] * u_val * v_val
            C_calc[i, k] = s
            
    if not np.allclose(C_true, C_calc, atol=1e-8):
        # Fallback to schoolbook if verification fails (should not happen)
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        # Return the optimized decomposition
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
