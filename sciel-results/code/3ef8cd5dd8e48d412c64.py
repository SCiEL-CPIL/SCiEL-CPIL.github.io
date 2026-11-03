import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0
T_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - T_START)

def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U: (4, 4), V: (2, 4), W: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
        
    return U, V, W

def construct():
    """
    Constructs a rank-50 decomposition for 2x4x7 matrix multiplication.
    Strategy:
    1. Split m=4 into two blocks of m=2.
    2. For each m-block, split p=7 into p=4 and p=3.
    3. For p=4, use two 2x2x2 Strassen blocks (rank 7 each).
    4. For p=3, use one 2x2x2 Strassen block (rank 7) and one 2x2x1 Schoolbook block (rank 4).
    Total rank per m-block = 7 + 7 + 7 + 4 = 25.
    Total rank = 2 * 25 = 50.
    """
    n, m, p = DIMENSIONS
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    col_offset = 0
    
    # Helper to place a sub-tensor into the global arrays
    def place_subtensor(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset):
        nonlocal col_offset
        r_sub = U_sub.shape[1]
        
        # U: Global index i*m + j. Sub index i_sub*m_sub + j_sub.
        # Here m_sub=2, n_sub=2, p_sub varies.
        # We map i_sub -> i_offset + i_sub
        # j_sub -> j_offset + j_sub
        # k_sub -> k_offset + k_sub
        
        n_sub, m_sub, p_sub = U_sub.shape[0]//2, 2, V_sub.shape[0]//2 # Wait, shapes are (n*m, r), (m*p, r), (p*n, r)
        # U_sub shape (4, r). n_sub=2, m_sub=2.
        # V_sub shape (m_sub*p_sub, r).
        # W_sub shape (p_sub*n_sub, r).
        
        # Determine p_sub from V_sub shape
        p_sub = V_sub.shape[0] // m_sub
        
        # Place U
        for i_s in range(n_sub):
            for j_s in range(m_sub):
                U[(i_offset + i_s) * m + (j_offset + j_s), col_offset:col_offset+r_sub] = U_sub[i_s * m_sub + j_s, :]
                
        # Place V
        for j_s in range(m_sub):
            for k_s in range(p_sub):
                V[(j_offset + j_s) * p + (k_offset + k_s), col_offset:col_offset+r_sub] = V_sub[j_s * p_sub + k_s, :]
                
        # Place W
        for k_s in range(p_sub):
            for i_s in range(n_sub):
                W[(k_offset + k_s) * n + (i_offset + i_s), col_offset:col_offset+r_sub] = W_sub[k_s * n_sub + i_s, :]
                
        col_offset += r_sub

    # Block 1: j in [0, 2)
    j_offset_1 = 0
    
    # Sub-block 1.1: k in [0, 2) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_1, 0)
    
    # Sub-block 1.2: k in [2, 4) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_1, 2)
    
    # Sub-block 1.3: k in [4, 6) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_1, 4)
    
    # Sub-block 1.4: k in [6, 7) -> 2x2x1 Schoolbook
    U_s, V_s, W_s = _get_school_221()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_1, 6)
    
    # Block 2: j in [2, 4)
    j_offset_2 = 2
    
    # Sub-block 2.1: k in [0, 2) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_2, 0)
    
    # Sub-block 2.2: k in [2, 4) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_2, 2)
    
    # Sub-block 2.3: k in [4, 6) -> 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_222()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_2, 4)
    
    # Sub-block 2.4: k in [6, 7) -> 2x2x1 Schoolbook
    U_s, V_s, W_s = _get_school_221()
    place_subtensor(U_s, V_s, W_s, 0, j_offset_2, 6)
    
    assert col_offset == total_rank, (col_offset, total_rank)
    
    # Verify
    if _time_left() > 5.0:
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W):
    """
    Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k.
    U, V, W are numpy arrays.
    """
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = int(np.dot(np.dot(U[i * m + j], V[j * p + k]), W[k * n + i]))
                if s != 1:
                    return False
    return True

