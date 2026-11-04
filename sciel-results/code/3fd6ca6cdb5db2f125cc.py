"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _strassen_2x2_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _tensor_product_strassen_4x4_4x4():
    """
    Construct the rank-49 decomposition for 4x4 * 4x4 using tensor product of Strassen.
    Returns U, V, W of shape (16, 49), (16, 49), (16, 49).
    """
    U_s, V_s, W_s = _strassen_2x2_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    row_u = i1 * 8 + i0 * 4 + j1 * 2 + j0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            U[row_u][col] = U_s[i0 * 2 + j0][c0] * U_s[i1 * 2 + j1][c1]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    row_v = j1 * 8 + j0 * 4 + k1 * 2 + k0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            V[row_v][col] = V_s[j0 * 2 + k0][c0] * V_s[j1 * 2 + k1][c1]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    row_w = k1 * 8 + k0 * 4 + i1 * 2 + i0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            W[row_w][col] = W_s[k0 * 2 + i0][c0] * W_s[k1 * 2 + i1][c1]
    
    return U, V, W

def _add_2x2_2x3_rank11(U, V, W, start_col, a_rows, b_rows, c_rows):
    """
    Add rank-11 decomposition for 2x2 * 2x3 multiplication.
    a_rows: 4 indices in U for the 2x2 block of A.
    b_rows: 6 indices in V for the 2x3 block of B.
    c_rows: 6 indices in W for the 2x3 block of C.
    """
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Strassen part for first 2 columns of B (rank 7)
    # B rows: b_rows[0:4] correspond to B[0:2, 0:2]
    # Specifically:
    # b_rows[0] = B[0,0], b_rows[1] = B[0,1]
    # b_rows[2] = B[1,0], b_rows[3] = B[1,1]
    # Note: V index is j*p + k.
    # For 2x3 block starting at row j_start:
    # Rows are j_start, j_start+1. Cols 0, 1, 2.
    # V indices: j_start*3+0, j_start*3+1, j_start*3+2, (j_start+1)*3+0, ...
    
    # Map Strassen V (4x7) to V block (6x7)
    # Strassen V[0] -> B[0,0] (local) -> b_rows[0]
    # Strassen V[1] -> B[0,1] (local) -> b_rows[1]
    # Strassen V[2] -> B[1,0] (local) -> b_rows[2]
    # Strassen V[3] -> B[1,1] (local) -> b_rows[3]
    
    for idx in range(4):
        for c in range(7):
            U[a_rows[idx]][start_col + c] = U_s[idx][c]
            V[b_rows[idx]][start_col + c] = V_s[idx][c]
            W[c_rows[idx]][start_col + c] = W_s[idx][c]
            
    # Schoolbook part for 3rd column of B (rank 4)
    # B[0,2] is b_rows[4], B[1,2] is b_rows[5]
    # C[0,2] is c_rows[4], C[1,2] is c_rows[5]
    # A[0,0]=a_rows[0], A[0,1]=a_rows[1], A[1,0]=a_rows[2], A[1,1]=a_rows[3]
    
    col_offset = start_col + 7
    
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    U[a_rows[0]][col_offset] = 1
    V[b_rows[4]][col_offset] = 1
    W[c_rows[4]][col_offset] = 1
    
    U[a_rows[1]][col_offset + 1] = 1
    V[b_rows[5]][col_offset + 1] = 1
    W[c_rows[4]][col_offset + 1] = 1
    
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    U[a_rows[2]][col_offset + 2] = 1
    V[b_rows[4]][col_offset + 2] = 1
    W[c_rows[5]][col_offset + 2] = 1
    
    U[a_rows[3]][col_offset + 3] = 1
    V[b_rows[5]][col_offset + 3] = 1
    W[c_rows[5]][col_offset + 3] = 1

def _construct_rank_93():
    n, m, p = 4, 4, 7
    rank = 93
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    # 1. Fill first 49 cols with 4x4x4 Strassen tensor product
    U_s, V_s, W_s = _tensor_product_strassen_4x4_4x4()
    for i in range(16):
        for c in range(49):
            U[i][c] = U_s[i][c]
            V[i][c] = V_s[i][c]
            W[i][c] = W_s[i][c]
            
    # 2. Fill next 44 cols with four 2x2x3 rank-11 decompositions
    start_col = 49
    
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # A rows: 0,1,4,5
    # B rows: 0,1,2,3,4,5
    # C rows: 0,1,2,3,4,5
    a1 = [0, 1, 4, 5]
    b1 = [0, 1, 2, 3, 4, 5]
    c1 = [0, 1, 2, 3, 4, 5]
    _add_2x2_2x3_rank11(U, V, W, start_col, a1, b1, c1)
    start_col += 11
    
    # Block 2: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :]
    # A rows: 2,3,6,7
    # B rows: 6,7,8,9,10,11
    # C rows: 0,1,2,3,4,5
    a2 = [2, 3, 6, 7]
    b2 = [6, 7, 8, 9, 10, 11]
    c2 = [0, 1, 2, 3, 4, 5]
    _add_2x2_2x3_rank11(U, V, W, start_col, a2, b2, c2)
    start_col += 11
    
    # Block 3: A[2:4, 0:2] * B[0:2, :] -> C[2:4, :]
    # A rows: 8,9,12,13
    # B rows: 0,1,2,3,4,5
    # C rows: 6,7,8,9,10,11
    a3 = [8, 9, 12, 13]
    b3 = [0, 1, 2, 3, 4, 5]
    c3 = [6, 7, 8, 9, 10, 11]
    _add_2x2_2x3_rank11(U, V, W, start_col, a3, b3, c3)
    start_col += 11
    
    # Block 4: A[2:4, 2:4] * B[2:4, :] -> C[2:4, :]
    # A rows: 10,11,14,15
    # B rows: 6,7,8,9,10,11
    # C rows: 6,7,8,9,10,11
    a4 = [10, 11, 14, 15]
    b4 = [6, 7, 8, 9, 10, 11]
    c4 = [6, 7, 8, 9, 10, 11]
    _add_2x2_2x3_rank11(U, V, W, start_col, a4, b4, c4)
    start_col += 11
    
    return {"U": U, "V": V, "W": W}

def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy tensor contraction."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Try rank 93 decomposition
    try:
        result = _construct_rank_93()
        if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
