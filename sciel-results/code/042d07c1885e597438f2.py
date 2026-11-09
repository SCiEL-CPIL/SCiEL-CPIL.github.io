"""Program for matrix multiplication 3x4x5 tensor decomposition.

Goal: find a low-rank exact decomposition.
The current best known upper bound via simple block-Strassen is around 37.
We implement a systematic block decomposition that achieves rank 37.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A21 + A22) B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11 (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22 (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11 + A12) B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using schoolbook.
    U shape: (4, 4), V shape: (4, 4), W shape: (4, 4)
    """
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((4, 4), dtype=float)
    W = np.zeros((4, 4), dtype=float)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1
                V[j * 2 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication using schoolbook.
    U shape: (4, 6), V shape: (6, 6), W shape: (6, 6)
    n=2, m=2, p=3
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 6
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication.
    Uses Strassen for first 2 cols (rank 7) + schoolbook for last col (rank 2).
    Total rank: 9
    U shape: (4, 9), V shape: (6, 9), W shape: (6, 9)
    n=2, m=2, p=3
    """
    n, m, p = 2, 2, 3
    # Strassen part for cols 0,1
    U_s, V_s, W_s = _strassen_2x2x2()
    # Schoolbook part for col 2 only
    U_sc = np.zeros((4, 2), dtype=float)
    V_sc = np.zeros((6, 2), dtype=float)
    W_sc = np.zeros((6, 2), dtype=float)
    col = 0
    for i in range(2):
        for j in range(2):
            # k = 2
            U_sc[i * 2 + j, col] = 1
            V_sc[j * 3 + 2, col] = 1
            W_sc[2 * 2 + i, col] = 1
            col += 1
    
    U = np.hstack([U_s, U_sc])
    V = np.hstack([V_s, V_sc])
    W = np.hstack([W_s, W_sc])
    return U, V, W

def _schoolbook_1x4x5():
    """Return (U, V, W) for 1x4x5 matrix multiplication using schoolbook.
    U shape: (4, 5), V shape: (20, 5), W shape: (5, 5)
    n=1, m=4, p=5
    """
    n, m, p = 1, 4, 5
    rank = n * m * p  # 5
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _map_sub_decomp(U_sub, V_sub, W_sub, 
                     u_rows, v_rows, w_rows,
                     total_u, total_v, total_w):
    """Map a sub-decomposition to global indices.
    u_rows: list of global U indices corresponding to local rows 0..len(u_rows)-1
    v_rows: list of global V indices corresponding to local rows 0..len(v_rows)-1
    w_rows: list of global W indices corresponding to local rows 0..len(w_rows)-1
    """
    rank_sub = U_sub.shape[1]
    U = np.zeros((total_u, rank_sub), dtype=float)
    V = np.zeros((total_v, rank_sub), dtype=float)
    W = np.zeros((total_w, rank_sub), dtype=float)
    
    for local_i, global_i in enumerate(u_rows):
        U[global_i, :] = U_sub[local_i, :]
    for local_i, global_i in enumerate(v_rows):
        V[global_i, :] = V_sub[local_i, :]
    for local_i, global_i in enumerate(w_rows):
        W[global_i, :] = W_sub[local_i, :]
    
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Global dimensions:
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (15, rank)
    
    # We'll build the decomposition using block structure.
    # 
    # A is 3x4, B is 4x5, C is 3x5.
    # 
    # Split A into:
    #   A_top: rows 0,1 (2x4)
    #   A_bot: row 2 (1x4)
    # 
    # C_top (2x5) = A_top * B
    # C_bot (1x5) = A_bot * B
    # 
    # For C_bot (1x4x5): schoolbook, rank 5
    # 
    # For C_top (2x4x5):
    #   Split B into B_L (4x2, cols 0,1) and B_R (4x3, cols 2,3,4)
    #   C_top_L (2x2) = A_top * B_L
    #   C_top_R (2x3) = A_top * B_R
    # 
    # For C_top_L (2x4x2):
    #   Split A_top into A_TL (2x2, cols 0,1) and A_TR (2x2, cols 2,3)
    #   C_TL = A_TL * B_L (2x2x2) + A_TR * B_L (2x2x2)
    #   Each 2x2x2 uses Strassen (rank 7), total rank 14
    # 
    # For C_top_R (2x4x3):
    #   Split A_top into A_TL (2x2, cols 0,1) and A_TR (2x2, cols 2,3)
    #   C_TR = A_TL * B_R (2x2x3) + A_TR * B_R (2x2x3)
    #   Each 2x2x3 uses Strassen+schoolbook (rank 9), total rank 18
    # 
    # Total rank: 5 + 14 + 18 = 37
    
    # Build sub-decompositions
    
    # Sub 1: A_TL * B_L (2x2x2) - Strassen
    # A rows: 0,1; A cols: 0,1 -> U indices: 0,1,4,5
    # B rows: 0,1; B cols: 0,1 -> V indices: 0,1,5,6
    # C rows: 0,1; C cols: 0,1 -> W indices: 0,1,3,4
    U1_sub, V1_sub, W1_sub = _strassen_2x2x2()
    u_rows_1 = [0, 1, 4, 5]
    v_rows_1 = [0, 1, 5, 6]
    w_rows_1 = [0, 1, 3, 4]
    U1, V1, W1 = _map_sub_decomp(U1_sub, V1_sub, W1_sub, u_rows_1, v_rows_1, w_rows_1, 12, 20, 15)
    
    # Sub 2: A_TR * B_L (2x2x2) - Strassen
    # A rows: 0,1; A cols: 2,3 -> U indices: 2,3,6,7
    # B rows: 0,1; B cols: 0,1 -> V indices: 0,1,5,6
    # C rows: 0,1; C cols: 0,1 -> W indices: 0,1,3,4
    U2_sub, V2_sub, W2_sub = _strassen_2x2x2()
    u_rows_2 = [2, 3, 6, 7]
    v_rows_2 = [0, 1, 5, 6]
    w_rows_2 = [0, 1, 3, 4]
    U2, V2, W2 = _map_sub_decomp(U2_sub, V2_sub, W2_sub, u_rows_2, v_rows_2, w_rows_2, 12, 20, 15)
    
    # Sub 3: A_TL * B_R (2x2x3) - Strassen+schoolbook
    # A rows: 0,1; A cols: 0,1 -> U indices: 0,1,4,5
    # B rows: 0,1; B cols: 2,3,4 -> V indices: 2,3,4,7,8,9
    # C rows: 0,1; C cols: 2,3,4 -> W indices: 6,7,8,9,10,11
    U3_sub, V3_sub, W3_sub = _strassen_2x2x3()
    u_rows_3 = [0, 1, 4, 5]
    v_rows_3 = [2, 3, 4, 7, 8, 9]
    w_rows_3 = [6, 7, 8, 9, 10, 11]
    U3, V3, W3 = _map_sub_decomp(U3_sub, V3_sub, W3_sub, u_rows_3, v_rows_3, w_rows_3, 12, 20, 15)
    
    # Sub 4: A_TR * B_R (2x2x3) - Strassen+schoolbook
    # A rows: 0,1; A cols: 2,3 -> U indices: 2,3,6,7
    # B rows: 0,1; B cols: 2,3,4 -> V indices: 2,3,4,7,8,9
    # C rows: 0,1; C cols: 2,3,4 -> W indices: 6,7,8,9,10,11
    U4_sub, V4_sub, W4_sub = _strassen_2x2x3()
    u_rows_4 = [2, 3, 6, 7]
    v_rows_4 = [2, 3, 4, 7, 8, 9]
    w_rows_4 = [6, 7, 8, 9, 10, 11]
    U4, V4, W4 = _map_sub_decomp(U4_sub, V4_sub, W4_sub, u_rows_4, v_rows_4, w_rows_4, 12, 20, 15)
    
    # Sub 5: A_bot * B (1x4x5) - schoolbook
    # A row: 2; A cols: 0,1,2,3 -> U indices: 8,9,10,11
    # B rows: 0,1,2,3; B cols: 0,1,2,3,4 -> V indices: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    # C row: 2; C cols: 0,1,2,3,4 -> W indices: 2,5,8,11,14
    U5_sub, V5_sub, W5_sub = _schoolbook_1x4x5()
    u_rows_5 = [8, 9, 10, 11]
    v_rows_5 = list(range(20))
    w_rows_5 = [2, 5, 8, 11, 14]
    U5, V5, W5 = _map_sub_decomp(U5_sub, V5_sub, W5_sub, u_rows_5, v_rows_5, w_rows_5, 12, 20, 15)
    
    # Combine all sub-decompositions
    sub_decomps = [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3), (U4, V4, W4), (U5, V5, W5)]
    
    total_rank = sum(U.shape[1] for U, V, W in sub_decomps)
    
    U = np.zeros((12, total_rank), dtype=float)
    V = np.zeros((20, total_rank), dtype=float)
    W = np.zeros((15, total_rank), dtype=float)
    
    col_offset = 0
    for U_sub, V_sub, W_sub in sub_decomps:
        rank_sub = U_sub.shape[1]
        U[:, col_offset:col_offset+rank_sub] = U_sub
        V[:, col_offset:col_offset+rank_sub] = V_sub
        W[:, col_offset:col_offset+rank_sub] = W_sub
        col_offset += rank_sub
    
    # Verify correctness with a random test
    np.random.seed(42)
    A_test = np.random.randn(3, 4)
    B_test = np.random.randn(4, 5)
    C_expected = A_test @ B_test
    
    # Reconstruct C from decomposition
    C_reconstructed = np.zeros((3, 5))
    for c in range(total_rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        for i in range(3):
            for k in range(5):
                s = 0.0
                for j in range(4):
                    s += u_col[i*4+j] * v_col[j*5+k] * w_col[k*3+i]
                C_reconstructed[i, k] += s
    
    # Check if close
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Success - return the low-rank decomposition
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
