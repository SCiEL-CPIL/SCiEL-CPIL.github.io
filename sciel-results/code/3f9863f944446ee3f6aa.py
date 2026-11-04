import time
import numpy as np

_DEADLINE = time.time() + 110

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Index convention:
    U[i, k] for A[i]
    V[j, k] for B[j]
    W[k', k] for C[k'] where C index is k_col * 2 + k_row
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[3, 1] = -1 # C00, C11
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1 # C01, C11
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[1, 3] = 1 # C01, C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = -1 # C10, C11
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1 # C10
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C00
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Outer Strassen coefficients for 4x4 blocks
    # Blocks: 00, 01, 10, 11
    # M1: A00+A11, B00+B11 -> C00+C11
    # M2: A10+A11, B00 -> -C00-C11
    # M3: A00, B01-B11 -> C01-C11
    # M4: A11, B10-B00 -> C01+C10
    # M5: A00+A01, B11 -> C10-C11
    # M6: A10-A00, B00+B01 -> C10
    # M7: A01-A11, B10+B11 -> -C00
    
    # Coefficients for A blocks (4 blocks)
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coefficients for B blocks (4 blocks)
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for C blocks (4 blocks)
    # C index in 4x4: k_col * 4 + k_row? No, W index is k*n + i.
    # For 4x4, n=4. W index = k_col * 4 + k_row.
    # Blocks:
    # C00: rows 0,1 cols 0,1
    # C01: rows 0,1 cols 2,3
    # C10: rows 2,3 cols 0,1
    # C11: rows 2,3 cols 2,3
    
    C_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [-1, 0, 0, -1],# M2 -> C00, C11
        [0, 1, 0, -1],# M3 -> C01, C11
        [0, 1, 1, 0], # M4 -> C01, C10
        [0, 0, 1, -1],# M5 -> C10, C11
        [0, 0, 1, 0], # M6 -> C10
        [-1, 0, 0, 0] # M7 -> C00
    ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for k in range(7):
        for l in range(7):
            col_idx = k * 7 + l
            
            # U: A
            # A block b has rows b_row*2..b_row*2+1, cols b_col*2..b_col*2+1
            for b in range(4):
                a_coeff = A_coeffs[k][b]
                if a_coeff == 0: continue
                b_row = b // 2
                b_col = b % 2
                # Local 2x2 indices in U2
                for li in range(4):
                    # li corresponds to local row/col in 2x2
                    # U2 row index: li
                    # Global row index: b_row*2 + (li//2), b_col*2 + (li%2)
                    # Wait, U2 index is i*m + j? No, U2 is for 2x2.
                    # U2 row 0: A00. Row 1: A01. Row 2: A10. Row 3: A11.
                    # So li=0 -> A00. li=1 -> A01. li=2 -> A10. li=3 -> A11.
                    # Global A index:
                    # Block b is at (b_row, b_col).
                    # Element li in block b is at global (b_row*2 + li//2, b_col*2 + li%2).
                    # Wait, li is the index in the 2x2 matrix.
                    # li=0: (0,0). li=1: (0,1). li=2: (1,0). li=3: (1,1).
                    g_row = b_row * 2 + (li // 2)
                    g_col = b_col * 2 + (li % 2)
                    g_idx = g_row * 4 + g_col
                    U4[g_idx, col_idx] = a_coeff * U2[li, l]
            
            # V: B
            for b in range(4):
                b_coeff = B_coeffs[k][b]
                if b_coeff == 0: continue
                b_row = b // 2
                b_col = b % 2
                for li in range(4):
                    g_row = b_row * 2 + (li // 2)
                    g_col = b_col * 2 + (li % 2)
                    g_idx = g_row * 4 + g_col
                    V4[g_idx, col_idx] = b_coeff * V2[li, l]
            
            # W: C
            for b in range(4):
                c_coeff = C_coeffs[k][b]
                if c_coeff == 0: continue
                b_row = b // 2
                b_col = b % 2
                for li in range(4):
                    g_row = b_row * 2 + (li // 2)
                    g_col = b_col * 2 + (li % 2)
                    # W index convention: k*n + i.
                    # Here n=4. k is col, i is row.
                    g_idx = g_col * 4 + g_row
                    W4[g_idx, col_idx] = c_coeff * W2[li, l]
                    
    return U4, V4, W4

def _schoolbook(n, m, p):
    """
    Returns U, V, W for nxm x m x p matrix multiplication.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                w_idx = k * n + i
                
                U[a_idx, idx] = 1.0
                V[b_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 9
    
    # Split B into B1 (4x4) and B2 (4x5)
    # C1 = A B1 (4x4)
    # C2 = A B2 (4x5)
    
    # Get 4x4x4 decomposition (Rank 49)
    U1, V1, W1 = _strassen_4x4()
    
    # Get 4x4x5 decomposition (Rank 80)
    U2, V2, W2 = _schoolbook(4, 4, 5)
    
    # Concatenate
    # U: (16, 49+80)
    U = np.hstack([U1, U2])
    
    # V: (36, 49+80)
    # V1 is (16, 49) for B1 (cols 0-3 of B, which are rows 0-15 of V)
    # V2 is (20, 80) for B2 (cols 4-8 of B, which are rows 16-35 of V)
    V = np.zeros((36, 129))
    V[:16, :49] = V1
    V[16:, 49:] = V2
    
    # W: (36, 49+80)
    # W1 is (16, 49) for C1 (cols 0-3 of C, which are rows 0-15 of W)
    # W2 is (20, 80) for C2 (cols 4-8 of C, which are rows 16-35 of W)
    W = np.zeros((36, 129))
    W[:16, :49] = W1
    W[16:, 49:] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
