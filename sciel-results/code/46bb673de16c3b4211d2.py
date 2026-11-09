"""Matrix multiplication 4,5,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _outer_product_2x1x2():
    """Rank-4 decomposition for (2x1)*(1x2) -> 2x2.
    U[i, r] for i in [0,2), V[k, r] for k in [0,2), W[k*2+i, r]
    C[i,k] = A[i]*B[k]
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    
    for i in range(2):
        for k in range(2):
            col = i*2 + k
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
    
    return U, V, W

def _block_decompose_4x5x6():
    """
    Decompose 4x5 * 5x6 using block structure.
    A: 4x5, split rows [2,2], cols [2,2,1]
    B: 5x6, split rows [2,2,1], cols [2,2,2]
    C: 4x6, split rows [2,2], cols [2,2,2]
    
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    Block types:
    - A[i,k] is 2x2, B[k,j] is 2x2: use Strassen (rank 7)
    - A[i,k] is 2x1, B[k,j] is 1x2: use outer product (rank 4)
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_o, V_o, W_o = _outer_product_2x1x2()
    
    # Block splits
    a_row_splits = [2, 2]
    a_col_splits = [2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    c_row_splits = [2, 2]
    c_col_splits = [2, 2, 2]
    
    # Precompute offsets
    a_row_off = [0]
    for s in a_row_splits[:-1]:
        a_row_off.append(a_row_off[-1] + s)
    a_col_off = [0]
    for s in a_col_splits[:-1]:
        a_col_off.append(a_col_off[-1] + s)
    b_row_off = [0]
    for s in b_row_splits[:-1]:
        b_row_off.append(b_row_off[-1] + s)
    b_col_off = [0]
    for s in b_col_splits[:-1]:
        b_col_off.append(b_col_off[-1] + s)
    c_row_off = [0]
    for s in c_row_splits[:-1]:
        c_row_off.append(c_row_off[-1] + s)
    c_col_off = [0]
    for s in c_col_splits[:-1]:
        c_col_off.append(c_col_off[-1] + s)
    
    # Count total rank
    total_rank = 0
    for i in range(2):
        for k in range(3):
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                if a_width == 2 and b_height == 2 and b_width == 2:
                    total_rank += 7
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    total_rank += 4
                else:
                    total_rank += a_width * b_height * b_width
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(4 * 5)]
    V = [[0] * total_rank for _ in range(5 * 6)]
    W = [[0] * total_rank for _ in range(6 * 4)]
    
    col = 0
    for i in range(2):
        for k in range(3):
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                
                a_i_start = a_row_off[i]
                a_k_start = a_col_off[k]
                b_k_start = b_row_off[k]
                b_j_start = b_col_off[j]
                c_i_start = c_row_off[i]
                c_j_start = c_col_off[j]
                
                if a_width == 2 and b_height == 2 and b_width == 2:
                    # Use Strassen for 2x2 * 2x2
                    for r in range(7):
                        # U: A[i,k] block, 2x2
                        for ii in range(2):
                            for jj in range(2):
                                val = U_s[ii*2+jj][r]
                                if val != 0:
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    U[u_row][col] += val
                        # V: B[k,j] block, 2x2
                        for kk in range(2):
                            for ll in range(2):
                                val = V_s[kk*2+ll][r]
                                if val != 0:
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    V[v_row][col] += val
                        # W: C[i,j] block, 2x2
                        for ll in range(2):
                            for mm in range(2):
                                val = W_s[ll*2+mm][r]
                                if val != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += val
                        col += 1
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    # Use outer product for 2x1 * 1x2
                    for r in range(4):
                        # U: A[i,k] block, 2x1
                        for ii in range(2):
                            val = U_o[ii][r]
                            if val != 0:
                                u_row = (a_i_start + ii) * 5 + a_k_start
                                U[u_row][col] += val
                        # V: B[k,j] block, 1x2
                        for ll in range(2):
                            val = V_o[ll][r]
                            if val != 0:
                                v_row = b_k_start * 6 + (b_j_start + ll)
                                V[v_row][col] += val
                        # W: C[i,j] block, 2x2
                        for ll in range(2):
                            for mm in range(2):
                                val = W_o[ll*2+mm][r]
                                if val != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += val
                        col += 1
                else:
                    # Fallback: schoolbook for this block
                    for ii in range(2):
                        for jj in range(a_width):
                            for kk in range(b_height):
                                for ll in range(b_width):
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + ii)
                                    U[u_row][col] = 1
                                    V[v_row][col] = 1
                                    W[w_row][col] = 1
                                    col += 1
    
    return {"U": U, "V": V, "W": W}, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    # Try block decomposition
    try:
        result, rank = _block_decompose_4x5x6()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
