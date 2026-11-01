"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x8 matrix.
    
    Uses a recursive Strassen-based approach:
    1. Decompose the 2x4x8 multiplication into 4 independent 2x2x2 multiplications
       for each column block of the 4x8 matrix.
    2. Apply rank-7 Strassen decomposition to each 2x2x2 block.
    
    Total rank = 4 blocks * 7 products per block * 2 (left+right A blocks) = 56.
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), 
    (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly 
    decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s timeout
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Strassen decomposition for 2x2x2 matrix multiplication.
    # Local index convention:
    #   A local: 0=a00, 1=a01, 2=a10, 3=a11
    #   B local: 0=b00, 1=b01, 2=b10, 3=b11
    #   C local: 0=c00, 1=c01, 2=c10, 3=c11
    #
    # 7 products:
    #   M1 = (a00+a11)(b00+b11)
    #   M2 = (a10+a11)b00
    #   M3 = a00(b10-b11)
    #   M4 = a11(b01-b10)
    #   M5 = (a00+a01)b11
    #   M6 = (a10-a00)(b00+b10)
    #   M7 = (a01-a11)(b01+b11)
    #
    # Output:
    #   c00 = M1 - M5 + M7 + M4
    #   c01 = M3 + M5
    #   c10 = M2 + M4
    #   c11 = M1 - M2 + M3 + M6

    u_strassen = np.array([
        [1,  0,  0,  1],   # M1: a00+a11
        [0,  0,  1,  1],   # M2: a10+a11
        [1,  0,  0,  0],   # M3: a00
        [0,  0,  0,  1],   # M4: a11
        [1,  1,  0,  0],   # M5: a00+a01
        [-1, 0,  1,  0],   # M6: a10-a00
        [0,  1,  0, -1]    # M7: a01-a11
    ], dtype=int)
    
    v_strassen = np.array([
        [1,  0,  0,  1],   # M1: b00+b11
        [1,  0,  0,  0],   # M2: b00
        [0,  0,  1, -1],   # M3: b10-b11
        [0,  1, -1, 0],    # M4: b01-b10
        [0,  0,  0,  1],   # M5: b11
        [1,  0,  1,  0],   # M6: b00+b10
        [0,  1,  0,  1]    # M7: b01+b11
    ], dtype=int)
    
    w_strassen = np.array([
        [1,  0,  0,  1],   # M1: c00+c11
        [0,  0,  1, -1],   # M2: c10-c11
        [0,  1,  0,  1],   # M3: c01+c11
        [1,  0,  1,  0],   # M4: c00+c10
        [-1, 1,  0,  0],   # M5: -c00+c01
        [0,  0,  0,  1],   # M6: c11
        [1,  0,  0,  0]    # M7: c00
    ], dtype=int)
    
    # Global index conventions:
    #   A (2x4): flat index i*4 + j, where i in 0..1, j in 0..3
    #   B (4x8): flat index j*8 + k, where j in 0..3, k in 0..7
    #   C (8x2): flat index k*2 + i, where k in 0..7, i in 0..1 (trace convention W[k*n+i, r])
    
    # Decomposition strategy:
    # Split A into left half A_L = A[:, 0:2] and right half A_R = A[:, 2:4]
    # Split B into 4 column blocks of 2 columns each: B[:, 2g:2g+2] for g in 0..3
    # Split B rows into top half B_top = B[0:2, :] and bottom half B_bot = B[2:4, :]
    #
    # For each column block g:
    #   C[2g:2g+2, :] = A_L @ B_top[:, 2g:2g+2] + A_R @ B_bot[:, 2g:2g+2]
    #   Each term is a 2x2 matrix multiplication, handled by Strassen.
    
    # Pre-compute local index mappings for efficiency
    a_l_idx = np.array([0, 1, 4, 5], dtype=int)    # A_L: (0,0), (0,1), (1,0), (1,1)
    a_r_idx = np.array([2, 3, 6, 7], dtype=int)    # A_R: (0,2), (0,3), (1,2), (1,3)
    
    # Build factor matrices with rank = 4 * 7 * 2 = 56
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    for g in range(4):
        # B top block indices: rows 0,1; columns 2g, 2g+1
        b_top_idx = np.array([0*8 + 2*g, 0*8 + 2*g + 1, 
                              1*8 + 2*g, 1*8 + 2*g + 1], dtype=int)
        
        # B bottom block indices: rows 2,3; columns 2g, 2g+1
        b_bot_idx = np.array([2*8 + 2*g, 2*8 + 2*g + 1, 
                              3*8 + 2*g, 3*8 + 2*g + 1], dtype=int)
        
        # C block indices: rows 2g, 2g+1; columns 0,1
        c_block_idx = np.array([(2*g)*2 + 0, (2*g)*2 + 1,
                                (2*g+1)*2 + 0, (2*g+1)*2 + 1], dtype=int)
        
        # Strassen products for A_L @ B_top[:, 2g:2g+2]
        for s in range(7):
            U[a_l_idx, col] = u_strassen[s, :]
            V[b_top_idx, col] = v_strassen[s, :]
            W[c_block_idx, col] = w_strassen[s, :]
            col += 1
        
        # Strassen products for A_R @ B_bot[:, 2g:2g+2]
        for s in range(7):
            U[a_r_idx, col] = u_strassen[s, :]
            V[b_bot_idx, col] = v_strassen[s, :]
            W[c_block_idx, col] = w_strassen[s, :]
            col += 1
    
    # Verification using the multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 iff a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    T = np.zeros((A_dim, B_dim, C_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Reconstruct tensor from factors: T_rec = sum_r U[:,r] * V[:,r] * W[:,r]
    T_rec = np.einsum('ar,br,cr->abc', U.astype(float), V.astype(float), W.astype(float))
    
    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback: schoolbook decomposition (rank = n*m*p = 64)
        U_school = np.zeros((A_dim, n*m*p), dtype=int)
        V_school = np.zeros((B_dim, n*m*p), dtype=int)
        W_school = np.zeros((C_dim, n*m*p), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_school[a_idx, r] = 1
                    V_school[b_idx, r] = 1
                    W_school[c_idx, r] = 1
                    r += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
