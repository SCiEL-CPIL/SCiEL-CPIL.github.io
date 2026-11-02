import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 exact tensor decomposition for 2x2 * 2x16 matrix multiplication.
    Uses Strassen (rank 7) for each of 8 independent 2x2 blocks.
    W convention: W[k*n + i, r]
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2   # 8 blocks
    R = num_blocks * 7    # rank 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen 2x2x2 coefficients
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B local: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # C local: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00       -> C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11*(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    
    U_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1},     # M7
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1},      # M7
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, C11
        {1: 1, 3: 1},      # M3 -> C01, C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> C00, C01
        {3: 1},            # M6 -> C11
        {0: 1},            # M7 -> C00
    ]
    
    for q in range(num_blocks):
        # B global indices for block q (columns 2q, 2q+1)
        # V index = row * p + col
        b_map = {
            0: 0 * p + 2 * q,       # B[0, 2q]
            1: 0 * p + (2 * q + 1), # B[0, 2q+1]
            2: 1 * p + 2 * q,       # B[1, 2q]
            3: 1 * p + (2 * q + 1), # B[1, 2q+1]
        }
        
        # C global indices for block q
        # W convention: k * n + i where k is output column, i is output row
        c_map = {
            0: (2 * q) * n + 0,     # C[0, 2q]   -> k=2q, i=0
            1: (2 * q + 1) * n + 0, # C[0, 2q+1] -> k=2q+1, i=0
            2: (2 * q) * n + 1,     # C[1, 2q]   -> k=2q, i=1
            3: (2 * q + 1) * n + 1, # C[1, 2q+1] -> k=2q+1, i=1
        }
        
        for s in range(7):
            r = q * 7 + s
            
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
                
            for b_local, coeff in V_coeffs[s].items():
                V[b_map[b_local], r] = coeff
                
            for c_local, coeff in W_coeffs[s].items():
                W[c_map[c_local], r] = coeff
    
    # Verify correctness
    np.random.seed(12345)
    valid = True
    for _ in range(3):
        A_t = np.random.randint(-5, 5, size=(n, m))
        B_t = np.random.randint(-5, 5, size=(m, p))
        C_t = A_t @ B_t  # shape (2, 16)
        
        A_flat = A_t.flatten()  # shape (4,)
        B_flat = B_t.flatten()  # shape (32,)
        
        # Compute: C_decomp = W @ (U^T A_flat) * (V^T B_flat)
        terms = (U.T @ A_flat) * (V.T @ B_flat)  # shape (56,)
        C_decomp = W @ terms  # shape (32,)
        
        # C_t is (2,16). W convention: index k*n+i = k*2+i
        # So C_decomp[k*2+i] should equal C_t[i, k]
        C_expected = C_t.T.flatten()  # shape (32,), index k*2+i
        
        if not np.array_equal(C_decomp, C_expected):
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook rank 64
        R_fb = n * m * p
        U = np.zeros((n * m, R_fb), dtype=int)
        V = np.zeros((m * p, R_fb), dtype=int)
        W = np.zeros((p * n, R_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
