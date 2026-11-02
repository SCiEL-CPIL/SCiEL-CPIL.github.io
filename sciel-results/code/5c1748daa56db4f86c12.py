import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b11-b12)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U coeffs for A (2x2)
    # M1: [1, 0; 0, 1]
    # M2: [0, 0; 1, 1]
    # M3: [1, 0; 0, 0]
    # M4: [0, 0; 0, 1]
    # M5: [1, 1; 0, 0]
    # M6: [-1, 0; 1, 0]  # (a21-a11) -> row 2 is 1, row 1 is -1
    # M7: [0, 1; 0, -1]  # (a12-a22) -> row 1 is 1, row 2 is -1
    
    u_strassen = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    # V coeffs for B (2x2)
    # M1: [1, 1; 1, 1]
    # M2: [1, 0; 0, 0]
    # M3: [0, 1; 0, -1] # (b21-b22) -> col 1 is 1, col 2 is -1? No, b21 is row 2 col 1.
    # Let's check indices.
    # b11 b12
    # b21 b22
    # M3 = a11 (b21 - b22).
    # V for M3: b21=1, b22=-1. Others 0.
    # [[0, 0], [1, -1]]
    # M4 = a22 (b11 - b12).
    # V for M4: b11=1, b12=-1.
    # [[1, -1], [0, 0]]
    # M5 = (a11+a12) b22.
    # V for M5: b22=1.
    # [[0, 0], [0, 1]]
    # M6 = (a21-a11) (b11+b12).
    # V for M6: b11=1, b12=1.
    # [[1, 1], [0, 0]]
    # M7 = (a12-a22) (b21+b22).
    # V for M7: b21=1, b22=1.
    # [[0, 0], [1, 1]]
    
    v_strassen = [
        [[1, 1], [1, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 0], [1, -1]], # M3
        [[1, -1], [0, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    # W coeffs for C (2x2)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W is (p*n, rank). In local 2x2, W is 4x1 per term.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in (k,i) order?
    # Trace convention W[k*n+i, r].
    # For 2x2 block at i0, k0.
    # Local k=0, i=0 -> global k0, i0 -> idx k0*n + i0
    # Local k=0, i=1 -> global k0, i0+1 -> idx k0*n + i0+1
    # Local k=1, i=0 -> global k0+1, i0 -> idx (k0+1)*n + i0
    # Local k=1, i=1 -> global k0+1, i0+1 -> idx (k0+1)*n + i0+1
    
    # Coefficients for each M in C11, C12, C21, C22
    # C11 (0,0): M1, M4, -M5, M7
    # C12 (0,1): M3, M5
    # C21 (1,0): M2, M4
    # C22 (1,1): M1, -M2, M3, M6
    
    w_strassen = []
    for m_idx in range(7):
        w = [0.0, 0.0, 0.0, 0.0]
        if m_idx == 0: # M1
            w[0] += 1; w[3] += 1
        if m_idx == 1: # M2
            w[2] += 1; w[3] -= 1
        if m_idx == 2: # M3
            w[1] += 1; w[3] += 1
        if m_idx == 3: # M4
            w[0] += 1; w[2] += 1
        if m_idx == 4: # M5
            w[0] -= 1; w[1] += 1
        if m_idx == 5: # M6
            w[3] += 1
        if m_idx == 6: # M7
            w[0] += 1
        
        w_strassen.append(w)
        
    def add_strassen_block(i0, j0, k0, col_offset):
        nonlocal col
        for m_idx in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    U[(i0+li)*m + (j0+lj), col] = u_strassen[m_idx][li][lj]
            # V
            for lj in range(2):
                for lk in range(2):
                    V[(j0+lj)*p + (k0+lk), col] = v_strassen[m_idx][lj][lk]
            # W
            for lk in range(2):
                for li in range(2):
                    # W index: (k0+lk)*n + (i0+li)
                    W[(k0+lk)*n + (i0+li), col] = w_strassen[m_idx][lk*2 + li]
            col += 1

    def add_schoolbook_block(i0, j0, k0, ni, nj, nk, col_offset):
        nonlocal col
        for li in range(ni):
            for lj in range(nj):
                for lk in range(nk):
                    U[(i0+li)*m + (j0+lj), col] = 1
                    V[(j0+lj)*p + (k0+lk), col] = 1
                    W[(k0+lk)*n + (i0+li), col] = 1
                    col += 1

    # Block 1: C11 (A_top B_left)
    # i in 0,1; j in 0,1,2,3; k in 0,1,2,3
    # Split j into 0,1 and 2,3. Split k into 0,1 and 2,3.
    # 4 Strassen blocks
    add_strassen_block(0, 0, 0, col) # i0=0, j0=0, k0=0
    add_strassen_block(0, 0, 2, col) # i0=0, j0=0, k0=2
    add_strassen_block(0, 2, 0, col) # i0=0, j0=2, k0=0
    add_strassen_block(0, 2, 2, col) # i0=0, j0=2, k0=2
    
    # Block 2: C12 (A_top B_right)
    # i in 0,1; j in 0,1,2,3; k in 4,5,6,7
    add_strassen_block(0, 0, 4, col)
    add_strassen_block(0, 0, 6, col)
    add_strassen_block(0, 2, 4, col)
    add_strassen_block(0, 2, 6, col)
    
    # Block 3: C21 (A_bot B_left)
    # i in 2; j in 0,1,2,3; k in 0,1,2,3
    # Schoolbook 1x4x4
    add_schoolbook_block(2, 0, 0, 1, 4, 4, col)
    
    # Block 4: C22 (A_bot B_right)
    # i in 2; j in 0,1,2,3; k in 4,5,6,7
    add_schoolbook_block(2, 0, 4, 1, 4, 4, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=None, m=None, p=None):
    if n is None:
        n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# Main entry point
def construct_entry():
    return construct()

# The task requires the function name to be construct()
# But the seed had construct() calling other things.
# I will keep the name construct() as the main one.
