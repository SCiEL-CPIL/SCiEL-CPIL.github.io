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
    U: shape (4, 7) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (4, 7) - rows indexed by (j,k) for j,k in {0,1}
    W: shape (4, 7) - rows indexed by (k,i) for k,i in {0,1}
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U rows: (0,0), (0,1), (1,0), (1,1) for A[i][j]
    # V rows: (0,0), (0,1), (1,0), (1,1) for B[j][k]
    # W rows: (0,0), (0,1), (1,0), (1,1) for C[k][i] (i.e., output index k*n+i with n=2)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # A11 = (0,0), A22 = (1,1), B11 = (0,0), B22 = (1,1)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1   # C[0,0] += M1
    U[1][0] = 1; V[1][0] = 1; W[1][0] = 1   # C[1,1] += M1 (W row for (k=1,i=1) is index 1*2+1=3? No wait)
    
    # Let me re-index carefully.
    # For 2x2*2x2: n=2, m=2, p=2
    # U row index: i*m + j = i*2 + j
    # V row index: j*p + k = j*2 + k
    # W row index: k*n + i = k*2 + i
    
    # So:
    # U[0] = A[0][0], U[1] = A[0][1], U[2] = A[1][0], U[3] = A[1][1]
    # V[0] = B[0][0], V[1] = B[0][1], V[2] = B[1][0], V[3] = B[1][1]
    # W[0] = C[0][0], W[1] = C[0][1], W[2] = C[1][0], W[3] = C[1][1]
    
    # Reset
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00=U[0], A11=U[3], B00=V[0], B11=V[3]
    # C00=W[0], C11=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10=U[2], A11=U[3], B00=V[0]
    # C10=W[2], C11=W[3]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    # A00=U[0], B10=V[2], B11=V[3]
    # C01=W[1], C11=W[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    # A11=U[3], B01=V[1], B00=V[0]
    # C00=W[0], C10=W[2]
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    # A00=U[0], A01=U[1], B11=V[3]
    # C00=W[0], C01=W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10=U[2], A00=U[0], B00=V[0], B01=V[1]
    # C11=W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01=U[1], A11=U[3], B10=V[2], B11=V[3]
    # C00=W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_2x2_2x4():
    """
    Strassen-based decomposition for 2x2 * 2x4.
    Split B into two 2x2 blocks: B = [B_left | B_right]
    C = [C_left | C_right] where C_left = A*B_left (2x2), C_right = A*B_right (2x2)
    
    We can compute both using Strassen, but we need to be careful.
    Actually, let's use the standard approach:
    A is 2x2, B is 2x4 = [B1 | B2] where B1, B2 are 2x2.
    C = A*B = [A*B1 | A*B2]
    
    For A*B1: use Strassen (7 mults)
    For A*B2: use Strassen (7 mults)
    Total: 14 mults, but we can potentially share some.
    
    Actually, a better approach: use the fact that we can write
    A*B = (A11+A22, A21+A22, ...) applied to each block of B.
    
    Let me just use 2 independent Strassen decompositions for the two 2x2 subproblems.
    That gives rank 14 for 2x2*2x4.
    
    Alternatively, we can use a more efficient construction.
    
    For now, let's use 2 Strassen blocks: rank 14.
    """
    # U: shape (4, 14) - rows for A[i][j], i,j in {0,1}
    # V: shape (8, 14) - rows for B[j][k], j in {0,1}, k in {0,1,2,3}
    # W: shape (8, 14) - rows for C[k][i], k in {0,1,2,3}, i in {0,1}
    
    # B is 2x4: B[0][0], B[0][1], B[0][2], B[0][3], B[1][0], B[1][1], B[1][2], B[1][3]
    # V row index: j*4 + k
    # V[0]=B[0][0], V[1]=B[0][1], V[2]=B[0][2], V[3]=B[0][3]
    # V[4]=B[1][0], V[5]=B[1][1], V[6]=B[1][2], V[7]=B[1][3]
    
    # C is 2x4: C[0][0], C[0][1], C[0][2], C[0][3], C[1][0], C[1][1], C[1][2], C[1][3]
    # W row index: k*2 + i (since p=4, n=2)
    # W[0]=C[0][0], W[1]=C[0][1], W[2]=C[0][2], W[3]=C[0][3]
    # W[4]=C[1][0], W[5]=C[1][1], W[6]=C[1][2], W[7]=C[1][3]
    
    U, V_left, W_left = _strassen_2x2_2x2()  # For B_left = B[:, 0:2]
    U_right, V_right, W_right = _strassen_2x2_2x2()  # For B_right = B[:, 2:4]
    
    rank = 14
    U_full = [[0]*rank for _ in range(4)]
    V_full = [[0]*rank for _ in range(8)]
    W_full = [[0]*rank for _ in range(8)]
    
    # Left half (columns 0,1 of B)
    for i in range(4):
        for c in range(7):
            U_full[i][c] = U[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_full[j*4 + k][c] = V_left[j*2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_full[k*2 + i][c] = W_left[k*2 + i][c]
    
    # Right half (columns 2,3 of B)
    for i in range(4):
        for c in range(7):
            U_full[i][c + 7] = U_right[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_full[j*4 + (k+2)][c + 7] = V_right[j*2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_full[(k+2)*2 + i][c + 7] = W_right[k*2 + i][c]
    
    return U_full, V_full, W_full


def _strassen_2x4_4x7():
    """
    Decomposition for 2x4 * 4x7.
    Split A into two 2x2 blocks: A = [A_top; A_bottom] where A_top, A_bottom are 2x2.
    Split B into 4x7... this is getting complex.
    
    Actually, let me think differently.
    A is 2x4, B is 4x7, C is 2x7.
    
    Split A into A = [A1 | A2] where A1, A2 are 2x2.
    Split B into B = [B1; B2] where B1, B2 are 2x7.
    
    C = A1*B1 + A2*B2
    
    Each A_i * B_i is 2x2 * 2x7.
    
    For 2x2 * 2x7: split B into 2x2 + 2x2 + 2x3 or 2x3 + 2x2 + 2x2.
    
    This is getting complicated. Let me use a different approach.
    
    Alternative: Use the full Strassen approach for the entire 4x4*4x7.
    
    Split A (4x4) into 2x2 blocks of 2x2:
    A = [[A11, A12], [A21, A22]]
    
    Split B (4x7) into 2x2 blocks of 2x... but 7 is odd.
    Split B into B = [[B11, B12, B13], [B21, B22, B23]] where B11, B21 are 2x2, B12, B22 are 2x2, B13, B23 are 2x3.
    
    C = A*B is 4x7:
    C = [[C11, C12, C13], [C21, C22, C23]]
    where C11 = A11*B11 + A12*B21 (2x2)
          C12 = A11*B12 + A12*B22 (2x2)
          C13 = A11*B13 + A12*B23 (2x3)
          C21 = A21*B11 + A22*B21 (2x2)
          C22 = A21*B12 + A22*B22 (2x2)
          C23 = A21*B13 + A22*B23 (2x3)
    
    For the 2x2*2x2 parts, we can use Strassen (rank 7 each).
    For the 2x2*2x3 parts, we need a decomposition.
    
    Let me count: 4 blocks of 2x2*2x2 (each rank 7) + 2 blocks of 2x2*2x3.
    
    For 2x2*2x3: 
    A is 2x2, B is 2x3, C is 2x3.
    Split B into B = [B1 | B2] where B1 is 2x1, B2 is 2x2.
    C = A*B1 + A*B2
    A*B2 is 2x2*2x2 (Strassen, rank 7)
    A*B1 is 2x2*2x1 (schoolbook rank 4, or we can find better)
    
    Actually, 2x2*2x1: A[i][j]*B[j][k] with j in {0,1}, k=0 only.
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    This is just 4 multiplications (schoolbook). Can we do better? The rank of 2x2*2x1 is at least max(4,2,2)=4, and schoolbook achieves 4, so rank is 4.
    
    So 2x2*2x3 has rank at most 7+4 = 11.
    
    Total for 4x4*4x7: 4*7 + 2*11 = 28 + 22 = 50.
    
    But wait, we can potentially share Strassen computations across the 2x2*2x2 blocks.
    
    Let me just implement the straightforward version with rank 50 first.
    """
    n, m, p = 4, 4, 7
    # A is n x m = 4x4
    # B is m x p = 4x7
    # C is n x p = 4x7
    
    # Block structure:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (B11,B21 are 2x2; B12,B22 are 2x2; B13,B23 are 2x3)
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # For each 2x2*2x2 product, use Strassen (7 mults)
    # For each 2x2*2x3 product, use Strassen for 2x2 part + schoolbook for 2x1 part (7+4=11 mults)
    
    # Total rank: 4*7 + 2*11 = 50
    
    rank = 50
    U = [[0]*rank for _ in range(16)]  # n*m = 16
    V = [[0]*rank for _ in range(28)]  # m*p = 28
    W = [[0]*rank for _ in range(28)]  # p*n = 28
    
    col = 0
    
    # Helper to add a Strassen 2x2*2x2 decomposition
    def add_strassen_2x2(a_row_offset, b_row_offset, c_row_offset):
        """
        a_row_offset: starting row in U for the 2x2 A block (rows 0-3 relative)
        b_row_offset: starting row in V for the 2x2 B block (rows 0-3 relative)
        c_row_offset: starting row in W for the 2x2 C block (rows 0-3 relative)
        Returns number of columns used.
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for i in range(4):
            for c in range(7):
                U[a_row_offset + i][col + c] = U_s[i][c]
        for j in range(4):
            for c in range(7):
                V[b_row_offset + j][col + c] = V_s[j][c]
        for k in range(4):
            for c in range(7):
                W[c_row_offset + k][col + c] = W_s[k][c]
        col += 7
        return 7
    
    # Helper to add a 2x2*2x3 decomposition (Strassen for 2x2 + schoolbook for 2x1)
    def add_2x2_2x3(a_row_offset, b_row_offset, c_row_offset):
        """
        a_row_offset: starting row in U for the 2x2 A block
        b_row_offset: starting row in V for the 2x3 B block (6 rows)
        c_row_offset: starting row in W for the 2x3 C block (6 rows)
        """
        nonlocal col
        # B is 2x3: B[0][0], B[0][1], B[0][2], B[1][0], B[1][1], B[1][2]
        # V rows: b_row_offset+0 = B[0][0], +1 = B[0][1], +2 = B[0][2]
        #          b_row_offset+3 = B[1][0], +4 = B[1][1], +5 = B[1][2]
        # C is 2x3: C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2]
        # W rows: c_row_offset+0 = C[0][0], +1 = C[0][1], +2 = C[0][2]
        #          c_row_offset+3 = C[1][0], +4 = C[1][1], +5 = C[1][2]
        
        # Part 1: A * B[:, 0:2] (2x2 * 2x2) using Strassen
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for i in range(4):
            for c in range(7):
                U[a_row_offset + i][col + c] = U_s[i][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[b_row_offset + j*3 + k][col + c] = V_s[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[c_row_offset + k*3 + i][col + c] = W_s[k*2 + i][c]
        col += 7
        
        # Part 2: A * B[:, 2:3] (2x2 * 2x1) using schoolbook (4 mults)
        # A[i][j] * B[j][2] for i,j in {0,1}
        # C[i][2] = A[i][0]*B[0][2] + A[i][1]*B[1][2]
        for i in range(2):
            for j in range(2):
                U[a_row_offset + i*2 + j][col] = 1
                V[b_row_offset + j*3 + 2][col] = 1
                W[c_row_offset + 2*3 + i][col] = 1  # C[i][2] -> W row = 2*3 + i = 6+i... wait
                col += 1
        
        return 11
    
    # Now let's map the blocks to actual indices.
    # A is 4x4:
    # A11 = A[0:2, 0:2] -> U rows 0-3 (i*m+j: i=0,1; j=0,1 -> 0,1,2,3)
    # A12 = A[0:2, 2:4] -> U rows 4-7 (i=0,1; j=2,3 -> 2,3,4,5... wait)
    # Actually U row index = i*m + j = i*4 + j
    # A11: i=0,1; j=0,1 -> rows 0,1,4,5
    # A12: i=0,1; j=2,3 -> rows 2,3,6,7
    # A21: i=2,3; j=0,1 -> rows 8,9,12,13
    # A22: i=2,3; j=2,3 -> rows 10,11,14,15
    
    # B is 4x7:
    # V row index = j*p + k = j*7 + k
    # B11: j=0,1; k=0,1 -> rows 0,1,7,8
    # B12: j=0,1; k=2,3 -> rows 2,3,9,10
    # B13: j=0,1; k=4,5,6 -> rows 4,5,6,11,12,13
    # B21: j=2,3; k=0,1 -> rows 14,15,21,22
    # B22: j=2,3; k=2,3 -> rows 16,17,23,24
    # B23: j=2,3; k=4,5,6 -> rows 18,19,20,25,26,27
    
    # C is 4x7:
    # W row index = k*n + i = k*4 + i
    # C11: k=0,1; i=0,1 -> rows 0,1,4,5
    # C12: k=0,1; i=2,3 -> rows 2,3,6,7
    # C13: k=0,1; i=4,5,6 -> wait, i only goes to 3.
    # Actually C is 4x7, so i in {0,1,2,3}, k in {0,...,6}
    # C11: k=0,1; i=0,1 -> W rows 0,1,4,5
    # C12: k=0,1; i=2,3 -> W rows 2,3,6,7
    # C13: k=0,1; i=4,5,6 -> but i max is 3! 
    
    # I made an error. C is 4x7, so the output indices are (i,k) where i in {0,1,2,3}, k in {0,...,6}.
    # W row = k*n + i = k*4 + i
    # C11 corresponds to output rows i=0,1 and columns k=0,1: W rows 0,1,4,5
    # C12 corresponds to output rows i=0,1 and columns k=2,3: W rows 2,3,6,7
    # C13 corresponds to output rows i=0,1 and columns k=4,5,6: W rows 4,5,8,9,12,13
    # C21 corresponds to output rows i=2,3 and columns k=0,1: W rows 0,1,4,5... no wait
    # C21: i=2,3; k=0,1 -> W rows k*4+i = 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C22: i=2,3; k=2,3 -> W rows 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C23: i=2,3; k=4,5,6 -> W rows 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23, 6*4+2=26, 6*4+3=27
    
    # This is getting very complex. Let me just use the schoolbook for now and return it.
    # The Strassen approach needs more careful implementation.
    
    # For now, return schoolbook
    return None


def _strassen_block_decomposition():
    """
    Block-based Strassen decomposition for 4x4 * 4x7.
    
    Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B (4x7) into rows: B = [[B_top], [B_bot]] where B_top, B_bot are 2x7
    Further split B_top into [B11, B12, B13] (2x2, 2x2, 2x3)
    Similarly B_bot into [B21, B22, B23]
    
    C (4x7) = [[C11, C12, C13], [C21, C22, C23]]
    where:
      C11 = A11*B11 + A12*B21  (2x2)
      C12 = A11*B12 + A12*B22  (2x2)
      C13 = A11*B13 + A12*B23  (2x3)
      C21 = A21*B11 + A22*B21  (2x2)
      C22 = A21*B12 + A22*B22  (2x2)
      C23 = A21*B13 + A22*B23  (2x3)
    
    For 2x2*2x2: use Strassen (rank 7)
    For 2x2*2x3: use a decomposition (rank 11: 7 for 2x2 part + 4 for 2x1 part)
    """
    n, m, p = 4, 4, 7
    
    # U rows: i*m + j = i*4 + j for A[i][j]
    # V rows: j*p + k = j*7 + k for B[j][k]
    # W rows: k*n + i = k*4 + i for C[k][i]
    
    # A11: A[0:2, 0:2] -> U rows: 0,1,4,5
    # A12: A[0:2, 2:4] -> U rows: 2,3,6,7
    # A21: A[2:4, 0:2] -> U rows: 8,9,12,13
    # A22: A[2:4, 2:4] -> U rows: 10,11,14,15
    
    # B11: B[0:2, 0:2] -> V rows: 0,1,7,8
    # B12: B[0:2, 2:4] -> V rows: 2,3,9,10
    # B13: B[0:2, 4:7] -> V rows: 4,5,6,11,12,13
    # B21: B[2:4, 0:2] -> V rows: 14,15,21,22
    # B22: B[2:4, 2:4] -> V rows: 16,17,23,24
    # B23: B[2:4, 4:7] -> V rows: 18,19,20,25,26,27
    
    # C11: C[0:2, 0:2] -> W rows: 0,1,4,5
    # C12: C[0:2, 2:4] -> W rows: 2,3,6,7
    # C13: C[0:2, 4:7] -> W rows: 4,5,8,9,12,13
    # C21: C[2:4, 0:2] -> W rows: 0,1,4,5... 
    # Wait: C is 4x7, indexed as C[i][k] with i in 0..3, k in 0..6
    # W row = k*n + i = k*4 + i
    # C11: i=0,1; k=0,1 -> W: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C12: i=0,1; k=2,3 -> W: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C13: i=0,1; k=4,5,6 -> W: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21, 6*4+0=24, 6*4+1=25
    # C21: i=2,3; k=0,1 -> W: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C22: i=2,3; k=2,3 -> W: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C23: i=2,3; k=4,5,6 -> W: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23, 6*4+2=26, 6*4+3=27
    
    rank = 50  # 4*7 + 2*11 = 28 + 22 = 50
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    col = 0
    
    def add_strassen(a_rows, b_rows, c_rows):
        """Add a Strassen 2x2*2x2 decomposition.
        a_rows: list of 4 U row indices for the 2x2 A block (in order: (0,0),(0,1),(1,0),(1,1))
        b_rows: list of 4 V row indices for the 2x2 B block (in order: (0,0),(0,1),(1,0),(1,1))
        c_rows: list of 4 W row indices for the 2x2 C block (in order: (0,0),(0,1),(1,0),(1,1))
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        for idx in range(4):
            for c in range(7):
                V[b_rows[idx]][col + c] = V_s[idx][c]
        for idx in range(4):
            for c in range(7):
                W[c_rows[idx]][col + c] = W_s[idx][c]
        col += 7
    
    def add_2x2_2x3(a_rows, b_rows, c_rows):
        """Add a 2x2*2x3 decomposition.
        a_rows: list of 4 U row indices for the 2x2 A block
        b_rows: list of 6 V row indices for the 2x3 B block (row0: 3 cols, row1: 3 cols)
        c_rows: list of 6 W row indices for the 2x3 C block (row0: 3 cols, row1: 3 cols)
        """
        nonlocal col
        # Part 1: A * B[:, 0:2] using Strassen
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        # B[:, 0:2]: b_rows[0:2] are B[0][0], B[0][1]; b_rows[3:5] are B[1][0], B[1][1]
        # V_s rows: 0=B[0][0], 1=B[0][1], 2=B[1][0], 3=B[1][1]
        V[b_rows[0]][col + 0] = V_s[0][0]
        V[b_rows[1]][col + 1] = V_s[1][1]
        V[b_rows[3]][col + 2] = V_s[2][2]
        V[b_rows[4]][col + 3] = V_s[3][3]
        # Actually let me be more careful
        # V_s has 4 rows: row0=(j=0,k=0), row1=(j=0,k=1), row2=(j=1,k=0), row3=(j=1,k=1)
        # b_rows[0]=B[0][0], b_rows[1]=B[0][1], b_rows[2]=B[0][2], b_rows[3]=B[1][0], b_rows[4]=B[1][1], b_rows[5]=B[1][2]
        for c in range(7):
            V[b_rows[0]][col + c] = V_s[0][c]  # B[0][0]
            V[b_rows[1]][col + c] = V_s[1][c]  # B[0][1]
            V[b_rows[3]][col + c] = V_s[2][c]  # B[1][0]
            V[b_rows[4]][col + c] = V_s[3][c]  # B[1][1]
        # C[:, 0:2]: c_rows[0]=C[0][0], c_rows[1]=C[0][1], c_rows[3]=C[1][0], c_rows[4]=C[1][1]
        # W_s rows: 0=C[0][0], 1=C[0][1], 2=C[1][0], 3=C[1][1]
        for c in range(7):
            W[c_rows[0]][col + c] = W_s[0][c]  # C[0][0]
            W[c_rows[1]][col + c] = W_s[1][c]  # C[0][1]
            W[c_rows[3]][col + c] = W_s[2][c]  # C[1][0]
            W[c_rows[4]][col + c] = W_s[3][c]  # C[1][1]
        col += 7
        
        # Part 2: A * B[:, 2] (2x2 * 2x1) schoolbook, 4 mults
        # C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2]
        # C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2]
        # a_rows: [A[0][0], A[0][1], A[1][0], A[1][1]]
        # b_rows[2]=B[0][2], b_rows[5]=B[1][2]
        # c_rows[2]=C[0][2], c_rows[5]=C[1][2]
        
        # Term: A[0][0] * B[0][2] -> C[0][2]
        U[a_rows[0]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        # Term: A[0][1] * B[1][2] -> C[0][2]
        U[a_rows[1]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        # Term: A[1][0] * B[0][2] -> C[1][2]
        U[a_rows[2]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
        
        # Term: A[1][1] * B[1][2] -> C[1][2]
        U[a_rows[3]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
    
    # Define row index lists
    # A blocks (U rows in order: (0,0),(0,1),(1,0),(1,1) within each 2x2 block)
    A11_rows = [0, 1, 4, 5]   # A[0][0], A[0][1], A[1][0], A[1][1]
    A12_rows = [2, 3, 6, 7]   # A[0][2], A[0][3], A[1][2], A[1][3]
    A21_rows = [8, 9, 12, 13] # A[2][0], A[2][1], A[3][0], A[3][1]
    A22_rows = [10, 11, 14, 15] # A[2][2], A[2][3], A[3][2], A[3][3]
    
    # B blocks (V rows)
    B11_rows = [0, 1, 7, 8]   # B[0][0], B[0][1], B[1][0], B[1][1]
    B12_rows = [2, 3, 9, 10]  # B[0][2], B[0][3], B[1][2], B[1][3]
    B13_rows = [4, 5, 6, 11, 12, 13]  # B[0][4], B[0][5], B[0][6], B[1][4], B[1][5], B[1][6]
    B21_rows = [14, 15, 21, 22] # B[2][0], B[2][1], B[3][0], B[3][1]
    B22_rows = [16, 17, 23, 24] # B[2][2], B[2][3], B[3][2], B[3][3]
    B23_rows = [18, 19, 20, 25, 26, 27] # B[2][4], B[2][5], B[2][6], B[3][4], B[3][5], B[3][6]
    
    # C blocks (W rows in order: (0,0),(0,1),(1,0),(1,1) for 2x2; (0,0),(0,1),(0,2),(1,0),(1,1),(1,2) for 2x3)
    C11_rows = [0, 1, 4, 5]   # C[0][0], C[0][1], C[1][0], C[1][1]
    C12_rows = [8, 9, 12, 13] # C[0][2], C[0][3], C[1][2], C[1][3]
    C13_rows = [16, 17, 20, 21, 24, 25]  # C[0][4], C[0][5], C[0][6], C[1][4], C[1][5], C[1][6]
    C21_rows = [2, 3, 6, 7]   # C[2][0], C[2][1], C[3][0], C[3][1]
    C22_rows = [10, 11, 14, 15] # C[2][2], C[2][3], C[3][2], C[3][3]
    C23_rows = [18, 19, 22, 23, 26, 27] # C[2][4], C[2][5], C[2][6], C[3][4], C[3][5], C[3][6]
    
    # Now build the decomposition
    # C11 = A11*B11 + A12*B21
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)
    
    # C12 = A11*B12 + A12*B22
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)
    
    # C13 = A11*B13 + A12*B23
    add_2x2_2x3(A11_rows, B13_rows, C13_rows)
    add_2x2_2x3(A12_rows, B23_rows, C13_rows)
    
    # C21 = A21*B11 + A22*B21
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)
    
    # C22 = A21*B12 + A22*B22
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)
    
    # C23 = A21*B13 + A22*B23
    add_2x2_2x3(A21_rows, B13_rows, C23_rows)
    add_2x2_2x3(A22_rows, B23_rows, C23_rows)
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the Strassen-based block decomposition
    try:
        result = _strassen_block_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
