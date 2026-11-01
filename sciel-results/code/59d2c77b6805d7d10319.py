"""
Matrix multiplication 2x2x15 tensor decomposition.
Uses block decomposition: 7 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (schoolbook, rank 4).
Total rank: 7*7 + 4 = 53.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) - indices i*2+j for A (i=0,1; j=0,1)
    V: (4, 7) - indices j*2+k for B (j=0,1; k=0,1)
    W: (4, 7) - indices k*2+i for C (k=0,1; i=0,1)
    
    The 7 intermediate products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b11)
    M4 = a11(b01-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    Output:
    c00 = M1 + M4 - M5 - M7
    c01 = M3 + M5
    c10 = M2 + M6
    c11 = M1 - M2 + M3 + M7
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00=1, a11=1
    U[0, 0] = 1.0  # a00
    U[3, 0] = 1.0  # a11
    # V: b00=1, b11=1
    V[0, 0] = 1.0  # b00
    V[3, 0] = 1.0  # b11
    # W: c00 += M1, c11 += M1
    W[0, 0] = 1.0  # c00 (k=0,i=0)
    W[3, 0] = 1.0  # c11 (k=1,i=1)
    
    # M2 = (a10+a11)b00
    # U: a10=1, a11=1
    U[2, 1] = 1.0  # a10
    U[3, 1] = 1.0  # a11
    # V: b00=1
    V[0, 1] = 1.0  # b00
    # W: c10 += M2, c11 -= M2
    W[2, 1] = 1.0  # c10 (k=0,i=1)
    W[3, 1] = -1.0 # c11 (k=1,i=1)
    
    # M3 = a00(b10-b11)
    # U: a00=1
    U[0, 2] = 1.0  # a00
    # V: b10=1, b11=-1
    V[2, 2] = 1.0  # b10
    V[3, 2] = -1.0 # b11
    # W: c01 += M3, c11 += M3
    W[1, 2] = 1.0  # c01 (k=1,i=0)
    W[3, 2] = 1.0  # c11 (k=1,i=1)
    
    # M4 = a11(b01-b00)
    # U: a11=1
    U[3, 3] = 1.0  # a11
    # V: b01=1, b00=-1
    V[1, 3] = 1.0  # b01
    V[0, 3] = -1.0 # b00
    # W: c00 -= M4, c10 += M4
    W[0, 3] = -1.0 # c00 (k=0,i=0)
    W[2, 3] = 1.0  # c10 (k=0,i=1)
    
    # M5 = (a00+a01)b11
    # U: a00=1, a01=1
    U[0, 4] = 1.0  # a00
    U[1, 4] = 1.0  # a01
    # V: b11=1
    V[3, 4] = 1.0  # b11
    # W: c00 += M5, c01 += M5
    W[0, 4] = 1.0  # c00 (k=0,i=0)
    W[1, 4] = 1.0  # c01 (k=1,i=0)
    
    # M6 = (a10-a00)(b00+b10)
    # U: a10=1, a00=-1
    U[2, 5] = 1.0  # a10
    U[0, 5] = -1.0 # a00
    # V: b00=1, b10=1
    V[0, 5] = 1.0  # b00
    V[2, 5] = 1.0  # b10
    # W: c10 += M6
    W[2, 5] = 1.0  # c10 (k=0,i=1)
    
    # M7 = (a01-a11)(b01+b11)
    # U: a01=1, a11=-1
    U[1, 6] = 1.0  # a01
    U[3, 6] = -1.0 # a11
    # V: b01=1, b11=1
    V[1, 6] = 1.0  # b01
    V[3, 6] = 1.0  # b11
    # W: c00 += M7, c11 += M7
    W[0, 6] = 1.0  # c00 (k=0,i=0)
    W[3, 6] = 1.0  # c11 (k=1,i=1)
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return the schoolbook decomposition for 2x2x1 matrix multiplication.
    U: (4, 4) - indices i*2+j for A
    V: (2, 4) - indices j for B (j=0,1)
    W: (2, 4) - indices i for C (i=0,1)
    """
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    
    return U, V, W

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Precompute decompositions
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    # Block structure: 7 blocks of size 2, 1 block of size 1
    # Total rank: 7*7 + 4 = 53
    total_rank = 7 * 7 + 4
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    # 7 blocks of 2x2x2 using Strassen
    for block in range(7):
        for r in range(7):
            # U: same for all k in this block (4 entries: a00, a01, a10, a11)
            U[0, col_offset] = U_s[0, r]  # a00
            U[1, col_offset] = U_s[1, r]  # a01
            U[2, col_offset] = U_s[2, r]  # a10
            U[3, col_offset] = U_s[3, r]  # a11
            
            # V: for j=0,1 and k=k_offset, k_offset+1
            # Global V index: j*p + k
            # Local V_s indices: 0->(j=0,k=0), 1->(j=0,k=1), 2->(j=1,k=0), 3->(j=1,k=1)
            V[0 * p + k_offset, col_offset] = V_s[0, r]
            V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
            V[1 * p + k_offset, col_offset] = V_s[2, r]
            V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
            
            # W: for k=k_offset, k_offset+1 and i=0,1
            # Global W index: k*n + i
            # Local W_s indices: 0->(k=0,i=0), 1->(k=1,i=0), 2->(k=0,i=1), 3->(k=1,i=1)
            W[k_offset * n + 0, col_offset] = W_s[0, r]
            W[(k_offset + 1) * n + 0, col_offset] = W_s[1, r]
            W[k_offset * n + 1, col_offset] = W_s[2, r]
            W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
            
            col_offset += 1
        k_offset += 2
    
    # 1 block of 2x2x1 using schoolbook
    for r in range(4):
        # U: a00, a01, a10, a11
        U[0, col_offset] = U_1[0, r]
        U[1, col_offset] = U_1[1, r]
        U[2, col_offset] = U_1[2, r]
        U[3, col_offset] = U_1[3, r]
        
        # V: b0, b1
        # Global V index: j*p + k
        V[0 * p + k_offset, col_offset] = V_1[0, r]
        V[1 * p + k_offset, col_offset] = V_1[1, r]
        
        # W: c0, c1
        # Global W index: k*n + i
        W[k_offset * n + 0, col_offset] = W_1[0, r]
        W[k_offset * n + 1, col_offset] = W_1[1, r]
        
        col_offset += 1
    k_offset += 1
    
    # Verify the decomposition
    tol = 1e-9
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0.0 and v_val != 0.0 and w_val != 0.0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook if Strassen block decomposition fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    return _construct()
# EVOLVE-BLOCK-END
