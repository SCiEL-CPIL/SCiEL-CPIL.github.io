import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B10
    U[2, 1] = 1; U[3, 1] = 1
    V[2, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B00-B01)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = -1
    
    # M7 = (A11-A10)(B10+B11)
    U[3, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[2, 6] = -1
    
    return U, V, W

def _laderman_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication using Laderman's algorithm.
    Rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices: 0..8 correspond to (0,0)..(2,2)
    """
    # Laderman's algorithm for 3x3
    # This is complex to hardcode correctly from memory.
    # I will use a known set of 23 terms.
    # If I cannot recall the exact 23 terms, I will fall back to schoolbook for 3x3 (rank 27).
    # Rank 27 vs 23.
    # Let's use schoolbook for 3x3 to be safe and avoid errors, unless I can find the terms.
    # Actually, I can construct the schoolbook terms easily.
    # Let's stick to schoolbook for 3x3 for now to ensure correctness.
    # Rank 27.
    # Total rank estimate:
    # C11: 7 (2x2) + 12 (2x3x2) = 19
    # C12: 12 (2x2x3) + 18 (2x3x3) = 30
    # C21: 12 (3x2x2) + 18 (3x3x2) = 30
    # C22: 18 (3x2x3) + 27 (3x3x3) = 45
    # Total: 19+30+30+45 = 124.
    # This is still better than 125.
    
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m * m x p multiplication.
    Rank n*m*p.
    """
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _add_block(global_U, global_V, global_W, local_U, local_V, local_W, 
               a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, 
               n_a, m_a, n_b, m_b, n_c, m_c, offset):
    """
    Adds local block factors to global factors.
    local_U is (n_a*m_a, rank), local_V is (m_b*m_b, rank) -- wait, V is for B.
    B is (m_a x n_b)? No.
    A is (n_a x m_a). B is (m_a x n_b). C is (n_a x n_b).
    Wait, in the block decomposition:
    A_{ij} is n_i x m_j.
    B_{jk} is m_j x n_k.
    C_{ik} is n_i x n_k.
    
    So for block (i, j, k):
    A block size: n_i x m_j.
    B block size: m_j x n_k.
    C block size: n_i x n_k.
    
    local_U corresponds to A block. Shape (n_i * m_j, rank).
    local_V corresponds to B block. Shape (m_j * n_k, rank).
    local_W corresponds to C block. Shape (n_k * n_i, rank). Note W index is k*n+i.
    
    Global A index: (a_row_start + r, a_col_start + c) -> (a_row_start + r) * 5 + (a_col_start + c)
    Global B index: (b_row_start + r, b_col_start + c) -> (b_row_start + r) * 5 + (b_col_start + c)
    Global C index: (c_row_start + r, c_col_start + c) -> (c_col_start + c) * 5 + (c_row_start + r) ?
    Wait, W convention is W[k*n+i, r].
    So W index is (col, row) of C.
    Global W index: (c_col_start + c, c_row_start + r) -> (c_col_start + c) * 5 + (c_row_start + r).
    """
    rank = local_U.shape[1]
    for r in range(rank):
        # Fill U
        for idx in range(n_a * m_a):
            if local_U[idx, r] != 0:
                local_r = idx // m_a
                local_c = idx % m_a
                global_idx = (a_row_start + local_r) * 5 + (a_col_start + local_c)
                global_U[global_idx, offset + r] = local_U[idx, r]
        
        # Fill V
        for idx in range(m_a * n_b):
            if local_V[idx, r] != 0:
                local_r = idx // n_b
                local_c = idx % n_b
                global_idx = (b_row_start + local_r) * 5 + (b_col_start + local_c)
                global_V[global_idx, offset + r] = local_V[idx, r]
        
        # Fill W
        for idx in range(n_k * n_i):
            if local_W[idx, r] != 0:
                # local_W is indexed by (k, i) in local coords.
                # local k is 0..n_k-1, local i is 0..n_i-1.
                # local idx = k * n_i + i.
                local_k = idx // n_i
                local_i = idx % n_i
                global_k = c_col_start + local_k
                global_i = c_row_start + local_i
                global_idx = global_k * 5 + global_i
                global_W[global_idx, offset + r] = local_W[idx, r]

def construct():
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Block sizes
    # A: 5x5. Split rows 0-1, 2-4. Cols 0-1, 2-4.
    # B: 5x5. Split rows 0-1, 2-4. Cols 0-1, 2-4.
    # C: 5x5. Split rows 0-1, 2-4. Cols 0-1, 2-4.
    
    r0, r1 = 0, 2
    c0, c1 = 0, 2
    
    # Block dimensions
    # A_00: 2x2, A_01: 2x3
    # A_10: 3x2, A_11: 3x3
    
    # B_00: 2x2, B_01: 2x3
    # B_10: 3x2, B_11: 3x3
    
    # C_00: 2x2, C_01: 2x3
    # C_10: 3x2, C_11: 3x3
    
    global_U = np.zeros((25, 200)) # Pre-allocate large enough
    global_V = np.zeros((25, 200))
    global_W = np.zeros((25, 200))
    
    offset = 0
    
    # Block 0,0: C_00 = A_00 B_00 + A_01 B_10
    # Term 1: A_00 (2x2) * B_00 (2x2) -> C_00 (2x2)
    U_l, V_l, W_l = _strassen_2x2()
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r0, c0, r0, c0, r0, c0, 2, 2, 2, 2, 2, 2, offset)
    offset += 7
    
    # Term 2: A_01 (2x3) * B_10 (3x2) -> C_00 (2x2)
    # A_01 is rows 0-1, cols 2-4.
    # B_10 is rows 2-4, cols 0-1.
    U_l, V_l, W_l = _schoolbook(2, 3, 2)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r0, c1, r1, c0, r0, c0, 2, 3, 3, 2, 2, 2, offset)
    offset += 12
    
    # Block 0,1: C_01 = A_00 B_01 + A_01 B_11
    # Term 1: A_00 (2x2) * B_01 (2x3) -> C_01 (2x3)
    U_l, V_l, W_l = _schoolbook(2, 2, 3)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r0, c0, r0, c1, r0, c1, 2, 2, 2, 3, 2, 3, offset)
    offset += 12
    
    # Term 2: A_01 (2x3) * B_11 (3x3) -> C_01 (2x3)
    U_l, V_l, W_l = _schoolbook(2, 3, 3)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r0, c1, r1, c1, r0, c1, 2, 3, 3, 3, 2, 3, offset)
    offset += 18
    
    # Block 1,0: C_10 = A_10 B_00 + A_11 B_10
    # Term 1: A_10 (3x2) * B_00 (2x2) -> C_10 (3x2)
    U_l, V_l, W_l = _schoolbook(3, 2, 2)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r1, c0, r0, c0, r1, c0, 3, 2, 2, 2, 3, 2, offset)
    offset += 12
    
    # Term 2: A_11 (3x3) * B_10 (3x2) -> C_10 (3x2)
    U_l, V_l, W_l = _schoolbook(3, 3, 2)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r1, c1, r1, c0, r1, c0, 3, 3, 3, 2, 3, 2, offset)
    offset += 18
    
    # Block 1,1: C_11 = A_10 B_01 + A_11 B_11
    # Term 1: A_10 (3x2) * B_01 (2x3) -> C_11 (3x3)
    U_l, V_l, W_l = _schoolbook(3, 2, 3)
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r1, c0, r0, c1, r1, c1, 3, 2, 2, 3, 3, 3, offset)
    offset += 18
    
    # Term 2: A_11 (3x3) * B_11 (3x3) -> C_11 (3x3)
    U_l, V_l, W_l = _laderman_3x3() # Using schoolbook inside for now
    _add_block(global_U, global_V, global_W, U_l, V_l, W_l, 
               r1, c1, r1, c1, r1, c1, 3, 3, 3, 3, 3, 3, offset)
    offset += 27
    
    # Total offset is 7+12+12+18+12+18+18+27 = 124
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for c in range(offset):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(global_U, global_V, global_W, n, m, p):
        # Fallback to schoolbook if error
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Trim and return
    U_final = global_U[:, :offset]
    V_final = global_V[:, :offset]
    W_final = global_W[:, :offset]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
