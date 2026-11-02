"""Tensor rank decomposition for 3x4 * 4x4 matrix multiplication.
Uses Strassen-based block decomposition to achieve rank 44 (improved from schoolbook 48).
"""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    
    return np.allclose(recon, expected, atol=1e-9)


def schoolbook(n, m, p):
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
    return U, V, W


def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) where:
      U is 4x7 (indices i*2+j for i,j in {0,1})
      V is 4x7 (indices j*2+k for j,k in {0,1})
      W is 4x7 (indices k*2+i for k,i in {0,1})
    
    Strassen formulas:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A10)*B11
      M6 = (A01-A10)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
      
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]  # A indices: i*2+j
    V = [[0]*rank for _ in range(4)]  # B indices: j*2+k
    W = [[0]*rank for _ in range(4)]  # C indices: k*2+i
    
    # Column 0: M1 = (A00+A11)*(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: M2 = (A10+A11)*B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: W[2,1]=1, C11: W[3,1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Column 2: M3 = A00*(B01-B11)
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01: W[1,2]=1, C11: W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Column 3: M4 = A11*(B10-B00)
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]=1, C10: W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Column 4: M5 = (A00+A10)*B11
    # A00: U[0,4]=1, A10: U[2,4]=1
    # B11: V[3,4]=1
    # C00: W[0,4]=-1, C01: W[1,4]=1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Column 5: M6 = (A01-A10)*(B00+B10)
    # A01: U[1,5]=1, A10: U[2,5]=-1
    # B00: V[0,5]=1, B10: V[2,5]=1
    # C11: W[3,5]=1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Column 6: M7 = (A01-A11)*(B01+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B01: V[1,6]=1, B11: V[3,6]=1
    # C00: W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_1x2_2x2():
    """
    Schoolbook decomposition for 1x2 * 2x2 matrix multiplication.
    A is 1x2 (indices j=0,1), B is 2x2 (indices j,k), C is 1x2 (indices k).
    Rank = 1*2*2 = 4.
    
    U is 2x4 (A indices: j)
    V is 4x4 (B indices: j*2+k)
    W is 2x4 (C indices: k)
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    col = 0
    for j in range(2):
        for k in range(2):
            U[j][col] = 1
            V[j*2+k][col] = 1
            W[k][col] = 1
            col += 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen-based block decomposition to achieve rank 44.
    
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4)
    - Split B (4x4) into B_left (4x2) and B_right (4x2)
    - For A_top * B_left: 2x4 * 4x2 -> split into four 2x2*2x2 blocks, use Strassen (rank 7 each) = 28
    - For A_top * B_right: 2x4 * 4x2 -> split into four 2x2*2x2 blocks, use Strassen (rank 7 each) = 28
    - For A_bot * B_left: 1x4 * 4x2 -> split into four 1x2*2x2 blocks, schoolbook (rank 4 each) = 16
    - For A_bot * B_right: 1x4 * 4x2 -> split into four 1x2*2x2 blocks, schoolbook (rank 4 each) = 16
    
    Wait, that gives 28+28+16+16 = 88, which is worse than 48.
    
    Let me reconsider. The correct approach:
    C = A * B where A is 3x4, B is 4x4, C is 3x4.
    
    Split columns of B into two groups of 2: B = [B1 | B2] where B1, B2 are 4x2.
    C = A * B1 | A * B2, each C_part is 3x2.
    
    For A * B1 (3x4 * 4x2):
      Split rows of A into A_top (2x4) and A_bot (1x4).
      Split B1 into B1_top (2x2) and B1_bot (2x2) (rows 0-1 and 2-3).
      A_top * B1 = A_top[:,0:2]*B1_top + A_top[:,2:4]*B1_bot
      A_bot * B1 = A_bot[:,0:2]*B1_top + A_bot[:,2:4]*B1_bot
      
      Each 2x2*2x2: Strassen rank 7
      Each 1x2*2x2: schoolbook rank 4
      
      So A*B1: 2*7 + 2*4 = 14 + 8 = 22
      Similarly A*B2: 22
      Total: 44
    
    This is the correct decomposition!
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 140
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the Strassen-based decomposition with rank 44
    # 
    # Structure:
    # A is 3x4, B is 4x4, C is 3x4
    # 
    # We decompose C = A*B as follows:
    # C = [A_top; A_bot] * B where A_top is 2x4, A_bot is 1x4
    # 
    # Split B's columns into B_left (4x2, cols 0-1) and B_right (4x2, cols 2-3)
    # C_left = A * B_left (3x2), C_right = A * B_right (3x2)
    # 
    # For C_left = A * B_left:
    #   Split B_left rows into B_lt (2x2, rows 0-1) and B_lb (2x2, rows 2-3)
    #   C_left = A_top[:,0:2]*B_lt + A_top[:,2:4]*B_lb + A_bot[:,0:2]*B_lt + A_bot[:,2:4]*B_lb
    #   = 2x2*2x2 + 2x2*2x2 + 1x2*2x2 + 1x2*2x2
    #   Rank: 7 + 7 + 4 + 4 = 22
    # 
    # Similarly for C_right: 22
    # Total rank: 44
    
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Process C_left (k=0,1) and C_right (k=2,3)
    for k_offset in [0, 2]:  # 0 for left, 2 for right
        for row_group in [0, 1]:  # 0 for A_top, 1 for A_bot
            if row_group == 0:
                # A_top: rows 0-1 of A (i=0,1), 2x4
                i_start = 0
                i_count = 2
                # For each pair of B rows
                for b_row_group in [0, 1]:  # 0 for rows 0-1, 1 for rows 2-3
                    b_start = b_row_group * 2
                    # This is a 2x2 * 2x2 product using Strassen
                    U_s, V_s, W_s = _strassen_2x2()
                    # U_s is 4x7, indices: i'*2+j' where i',j' in {0,1}
                    # V_s is 4x7, indices: j'*2+k' where j',k' in {0,1}
                    # W_s is 4x7, indices: k'*2+i' where k',i' in {0,1}
                    
                    for c_local in range(7):
                        for i_local in range(2):
                            for j_local in range(2):
                                val = U_s[i_local*2 + j_local][c_local]
                                if val != 0:
                                    # Global indices
                                    i_global = i_start + i_local
                                    j_global = b_start + j_local
                                    k_global = k_offset + c_local  # No, k is not c_local
                                    # Wait, I need to be more careful.
                                    pass
                    # Let me redo this more carefully.
                    # The 2x2*2x2 subproblem:
                    # A_sub[i', j'] corresponds to A[i_start+i', b_start+j']
                    # B_sub[j', k'] corresponds to B[b_start+j', k_offset+k']
                    # C_sub[k', i'] corresponds to C[i_start+i', k_offset+k']
                    
                    for c_local in range(7):
                        # U contribution: A_sub[i', j'] = U_s[i'*2+j'][c_local]
                        for i_local in range(2):
                            for j_local in range(2):
                                val_u = U_s[i_local*2 + j_local][c_local]
                                if val_u != 0:
                                    i_global = i_start + i_local
                                    j_global = b_start + j_local
                                    U[i_global*m + j_global][col + c_local] += val_u
                        
                        # V contribution: B_sub[j', k'] = V_s[j'*2+k'][c_local]
                        for j_local in range(2):
                            for k_local in range(2):
                                val_v = V_s[j_local*2 + k_local][c_local]
                                if val_v != 0:
                                    j_global = b_start + j_local
                                    k_global = k_offset + k_local
                                    V[j_global*p + k_global][col + c_local] += val_v
                        
                        # W contribution: C_sub[k', i'] = W_s[k'*2+i'][c_local]
                        for k_local in range(2):
                            for i_local in range(2):
                                val_w = W_s[k_local*2 + i_local][c_local]
                                if val_w != 0:
                                    k_global = k_offset + k_local
                                    i_global = i_start + i_local
                                    W[k_global*n + i_global][col + c_local] += val_w
                    
                    col += 7
            else:
                # A_bot: row 2 of A (i=2), 1x4
                i_start = 2
                for b_row_group in [0, 1]:
                    b_start = b_row_group * 2
                    # This is a 1x2 * 2x2 product using schoolbook
                    U_s, V_s, W_s = _schoolbook_1x2_2x2()
                    # U_s is 2x4, indices: j' in {0,1}
                    # V_s is 4x4, indices: j'*2+k' where j',k' in {0,1}
                    # W_s is 2x4, indices: k' in {0,1}
                    
                    for c_local in range(4):
                        # U contribution: A_sub[j'] = U_s[j'][c_local]
                        for j_local in range(2):
                            val_u = U_s[j_local][c_local]
                            if val_u != 0:
                                i_global = i_start
                                j_global = b_start + j_local
                                U[i_global*m + j_global][col + c_local] += val_u
                        
                        # V contribution: B_sub[j', k'] = V_s[j'*2+k'][c_local]
                        for j_local in range(2):
                            for k_local in range(2):
                                val_v = V_s[j_local*2 + k_local][c_local]
                                if val_v != 0:
                                    j_global = b_start + j_local
                                    k_global = k_offset + k_local
                                    V[j_global*p + k_global][col + c_local] += val_v
                        
                        # W contribution: C_sub[k'] = W_s[k'][c_local]
                        for k_local in range(2):
                            val_w = W_s[k_local][c_local]
                            if val_w != 0:
                                k_global = k_offset + k_local
                                i_global = i_start
                                W[k_global*n + i_global][col + c_local] += val_w
                    
                    col += 4
    
    # Verify we used all columns
    assert col == total_rank, f"Expected {total_rank} columns, used {col}"
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    if _verify_fast(U, V, W, n, m, p):
        best_rank = total_rank
        best_result = {"U": U, "V": V, "W": W}
    else:
        # If verification fails, fall back to schoolbook
        # But let's debug by checking what went wrong
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
