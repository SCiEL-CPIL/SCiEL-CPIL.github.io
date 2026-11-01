import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-18 decomposition.
    # It consists of:
    # 1. Strassen for cols 0-1 (rank 7)
    # 2. Strassen for cols 2-3 (rank 7)
    # 3. Schoolbook for col 4 (rank 4)
    
    rank = 18
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a Strassen block for a specific column offset in B
    # B is (m x p). We are processing B[:, col_offset : col_offset+2]
    # A is (n x m).
    # The result C is (n x p). We are filling C[:, col_offset : col_offset+2]
    
    def add_strassen(col_offset, u_offset, v_offset, w_offset):
        # Strassen 2x2x2 decomposition
        # U (4x7), V (4x7), W (4x7)
        # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Strassen matrices
        # M1 = (a+d)(e+f) -> c00, c11, c10, c01
        # M2 = (c+d)e -> c10
        # M3 = a(f-g) -> c01
        # M4 = d(h-g) -> c11
        # M5 = (a+b)h -> c00, c01
        # M6 = (c-a)(e+f) -> c10
        # M7 = (b-d)(g+h) -> c11
        
        # U (A factors)
        # M1: a+d -> [1, 0, 1, 0]
        # M2: c+d -> [0, 0, 1, 1]
        # M3: a   -> [1, 0, 0, 0] (times f-g, but U is just A part)
        # Actually, Strassen is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U (4x7)
        U_block = np.array([
            [1, 0, 1, 0, 1, -1, 0], # A11 (row 0)
            [0, 0, 0, 0, 1, 0, 1],  # A12 (row 1)
            [1, 1, 0, 0, 0, 1, 0],  # A21 (row 2)
            [1, 1, 0, 1, 0, 0, -1]  # A22 (row 3)
        ])
        
        # V (4x7) - Note B is (2x2) in this block, so m=2, p=2.
        # B indices: 0:B11, 1:B12, 2:B21, 3:B22
        V_block = np.array([
            [1, 1, 0, 0, 0, 1, 0], # B11
            [0, 0, -1, 1, 0, 0, 1],# B12
            [1, 0, 1, -1, 0, 1, 1],# B21
            [1, 0, 0, 0, 1, 0, 1]  # B22
        ])
        
        # W (4x7) - C indices: 0:C11, 1:C12, 2:C21, 3:C22
        W_block = np.array([
            [1, 0, 0, 0, 1, 0, 0], # C11
            [0, 0, 1, 0, 1, 0, 0], # C12
            [0, 1, 0, 0, 0, 1, 0], # C21
            [1, 0, 0, 1, 0, 0, 1]  # C22
        ])
        
        # Add to global matrices
        # U: A is (2x2). Rows 0-3.
        U[0:4, u_offset:u_offset+7] += U_block
        
        # V: B is (2x5). We are using cols col_offset, col_offset+1.
        # B indices in V: row = j*p + k.
        # j=0: k=col_offset, col_offset+1 -> rows col_offset, col_offset+1
        # j=1: k=col_offset, col_offset+1 -> rows p+col_offset, p+col_offset+1
        # V_block rows: 0:B11, 1:B12, 2:B21, 3:B22
        # So:
        # V_block row 0 (B11) -> V[row col_offset, :]
        # V_block row 1 (B12) -> V[row col_offset+1, :]
        # V_block row 2 (B21) -> V[row p+col_offset, :]
        # V_block row 3 (B22) -> V[row p+col_offset+1, :]
        
        V[col_offset, v_offset:v_offset+7] += V_block[0]
        V[col_offset+1, v_offset:v_offset+7] += V_block[1]
        V[p+col_offset, v_offset:v_offset+7] += V_block[2]
        V[p+col_offset+1, v_offset:v_offset+7] += V_block[3]
        
        # W: C is (2x5). We are using cols col_offset, col_offset+1.
        # W indices: row = k*n + i.
        # k=0: i=0,1 -> rows 0, 1
        # k=1: i=0,1 -> rows n, n+1
        # Wait, W convention is W[k*n+i, r].
        # C is (n x p). C[i, k].
        # So C[0, col_offset] is at W[0*n + 0] = W[0]
        # C[0, col_offset+1] is at W[0*n + 0]? No.
        # W index is k*n + i.
        # C[0, col_offset] -> k=col_offset, i=0 -> W[col_offset*n + 0]
        # C[0, col_offset+1] -> k=col_offset+1, i=0 -> W[(col_offset+1)*n + 0]
        # C[1, col_offset] -> k=col_offset, i=1 -> W[col_offset*n + 1]
        # C[1, col_offset+1] -> k=col_offset+1, i=1 -> W[(col_offset+1)*n + 1]
        
        # W_block rows: 0:C11, 1:C12, 2:C21, 3:C22
        # C11 is C[0, 0] relative to block. Global C[0, col_offset].
        # C12 is C[0, 1] relative. Global C[0, col_offset+1].
        # C21 is C[1, 0] relative. Global C[1, col_offset].
        # C22 is C[1, 1] relative. Global C[1, col_offset+1].
        
        W[col_offset*n + 0, w_offset:w_offset+7] += W_block[0]
        W[(col_offset+1)*n + 0, w_offset:w_offset+7] += W_block[1]
        W[col_offset*n + 1, w_offset:w_offset+7] += W_block[2]
        W[(col_offset+1)*n + 1, w_offset:w_offset+7] += W_block[3]

    # 1. Strassen for cols 0-1
    add_strassen(0, 0, 0, 0)
    
    # 2. Strassen for cols 2-3
    add_strassen(2, 7, 7, 7)
    
    # 3. Schoolbook for col 4
    # A (2x2) * B_col (2x1)
    # Terms: a*e, b*f, c*e, d*f
    # U: a, b, c, d
    # V: e, f, e, f
    # W: c0, c0, c1, c1
    
    u_offset = 14
    v_offset = 14
    w_offset = 14
    
    # Term 1: a * e -> c0
    U[0, u_offset] = 1
    V[4, v_offset] = 1 # B[0, 4] is row 0, col 4. Index 0*2 + 4 = 4.
    W[4*2 + 0, w_offset] = 1 # C[0, 4]. Index 4*2 + 0 = 8.
    
    # Term 2: b * f -> c0
    U[1, u_offset+1] = 1
    V[5, v_offset+1] = 1 # B[1, 4] is row 1, col 4. Index 1*2 + 4 = 5.
    W[4*2 + 0, w_offset+1] = 1
    
    # Term 3: c * e -> c1
    U[2, u_offset+2] = 1
    V[4, v_offset+2] = 1
    W[4*2 + 1, w_offset+2] = 1 # C[1, 4]. Index 4*2 + 1 = 9.
    
    # Term 4: d * f -> c1
    U[3, u_offset+3] = 1
    V[5, v_offset+3] = 1
    W[4*2 + 1, w_offset+3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
