import time
import numpy as np

def construct():
    t_start = time.time()
    deadline = 120.0
    
    n, m, p = 2, 2, 13
    
    # Use Strassen for 6 pairs of columns (12 columns) + schoolbook for 1 column
    # Total rank: 6*7 + 4 = 46
    
    rank = 6 * 7 + 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen block for columns (k, k')
    # A is 2x2: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B is 2x2: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # where B[0,0]=B_row0_col_k, B[0,1]=B_row0_col_kp, B[1,0]=B_row1_col_k, B[1,1]=B_row1_col_kp
    
    # Strassen multipliers:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U indices for A: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices for B: 0=B00(B_row0_colk), 1=B01(B_row0_colkp), 2=B10(B_row1_colk), 3=B11(B_row1_colkp)
    # W: distributes to C00, C01, C10, C11 which map to output positions
    
    # For columns k and k', the outputs are:
    # C[0,k] -> index k*n + 0 = k*2 + 0
    # C[0,k'] -> index k'*n + 0 = k'*2 + 0
    # C[1,k] -> index k*n + 1 = k*2 + 1
    # C[1,k'] -> index k'*n + 1 = k'*2 + 1
    
    def add_strassen_block(k, kp, col_start):
        """Add Strassen block for columns k and kp starting at column col_start"""
        # M1 = (A00+A11)*(B00+B11)
        U[0, col_start] = 1.0   # A00
        U[3, col_start] = 1.0   # A11
        V[0 * p + k, col_start] = 1.0    # B00 = B_row0_colk
        V[1 * p + kp, col_start] = 1.0   # B11 = B_row1_colkp
        W[k * n + 0, col_start] = 1.0    # C00
        W[kp * n + 0, col_start] = 1.0   # C01
        W[k * n + 1, col_start] = -1.0   # C10
        W[kp * n + 1, col_start] = 1.0   # C11
        col_start += 1
        
        # M2 = (A10+A11)*B00
        U[2, col_start] = 1.0   # A10
        U[3, col_start] = 1.0   # A11
        V[0 * p + k, col_start] = 1.0    # B00
        W[k * n + 1, col_start] = 1.0    # C10
        col_start += 1
        
        # M3 = A00*(B01-B11)
        U[0, col_start] = 1.0   # A00
        V[0 * p + kp, col_start] = 1.0   # B01 = B_row0_colkp
        V[1 * p + kp, col_start] = -1.0  # B11 = B_row1_colkp
        W[kp * n + 0, col_start] = 1.0   # C01
        W[kp * n + 1, col_start] = 1.0   # C11
        col_start += 1
        
        # M4 = A11*(B10-B00)
        U[3, col_start] = 1.0   # A11
        V[1 * p + k, col_start] = 1.0    # B10 = B_row1_colk
        V[0 * p + k, col_start] = -1.0   # B00
        W[k * n + 0, col_start] = 1.0    # C00
        W[k * n + 1, col_start] = 1.0    # C10
        col_start += 1
        
        # M5 = (A00+A01)*B11
        U[0, col_start] = 1.0   # A00
        U[1, col_start] = 1.0   # A01
        V[1 * p + kp, col_start] = 1.0   # B11
        W[k * n + 0, col_start] = -1.0   # C00
        W[kp * n + 0, col_start] = 1.0   # C01
        col_start += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, col_start] = 1.0   # A10
        U[0, col_start] = -1.0  # A00
        V[0 * p + k, col_start] = 1.0    # B00
        V[0 * p + kp, col_start] = 1.0   # B01
        W[kp * n + 1, col_start] = 1.0   # C11
        col_start += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, col_start] = 1.0   # A01
        U[3, col_start] = -1.0  # A11
        V[1 * p + k, col_start] = 1.0    # B10
        V[1 * p + kp, col_start] = 1.0   # B11
        W[k * n + 0, col_start] = 1.0    # C00
        W[k * n + 1, col_start] = -1.0   # C10
        col_start += 1
        
        return col_start
    
    # Add 6 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for pair in range(6):
        k = 2 * pair
        kp = 2 * pair + 1
        col = add_strassen_block(k, kp, col)
    
    # Schoolbook for column 12
    # C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    # C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    
    # Term 1: A[0,0]*B[0,12]
    U[0, col] = 1.0
    V[0 * p + 12, col] = 1.0
    W[12 * n + 0, col] = 1.0
    col += 1
    
    # Term 2: A[0,1]*B[1,12]
    U[1, col] = 1.0
    V[1 * p + 12, col] = 1.0
    W[12 * n + 0, col] = 1.0
    col += 1
    
    # Term 3: A[1,0]*B[0,12]
    U[2, col] = 1.0
    V[0 * p + 12, col] = 1.0
    W[12 * n + 1, col] = 1.0
    col += 1
    
    # Term 4: A[1,1]*B[1,12]
    U[3, col] = 1.0
    V[1 * p + 12, col] = 1.0
    W[12 * n + 1, col] = 1.0
    col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(recon, T, atol=1e-10):
        # Try to debug
        diff = np.abs(recon - T)
        max_diff = np.max(diff)
        raise RuntimeError(f"Decomposition verification failed, max diff: {max_diff}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
