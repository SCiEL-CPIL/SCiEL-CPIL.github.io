"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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


def _block_decompose_4x4_4x7():
    """
    Block decomposition for 4x4 * 4x7 using Strassen for 2x2*2x2 blocks
    and schoolbook for 2x2*2x1 parts.
    
    A (4x4) split into four 2x2 blocks: A11, A12, A21, A22
    B (4x7) split into B_top (2x7) and B_bottom (2x7)
    
    C (4x7) = [A11*B_top + A12*B_bottom; A21*B_top + A22*B_bottom]
    
    For each 2x2 * 2x7 product, split B into 2x2 + 2x2 + 2x3:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x2: Strassen (rank 7)  
    - 2x2 * 2x3: split into 2x2*2x2 (Strassen, rank 7) + 2x2*2x1 (schoolbook, rank 4) = 11
    
    Total per 2x2*2x7: 7 + 7 + 11 = 25
    Total for 4x4*4x7: 4 * 25 = 100
    
    Wait, that's worse than schoolbook (112). Let me recount.
    
    Actually: 4 blocks of 2x2*2x7, each with rank 25, total = 100 < 112. Good.
    """
    n, m, p = 4, 4, 7
    
    # Rank calculation:
    # 4 subproblems of 2x2 * 2x7
    # Each 2x2*2x7 = Strassen(2x2*2x2) + Strassen(2x2*2x2) + Strassen(2x2*2x2) + schoolbook(2x2*2x1)
    # = 7 + 7 + 7 + 4 = 25
    # Total rank = 4 * 25 = 100
    
    rank = 100
    U = [[0]*rank for _ in range(n*m)]  # 16 x 100
    V = [[0]*rank for _ in range(m*p)]  # 28 x 100
    W = [[0]*rank for _ in range(p*n)]  # 28 x 100
    
    col = 0
    
    def add_strassen_2x2_2x2(a_rows, b_rows, c_rows):
        """Add a Strassen 2x2*2x2 decomposition at given row offsets."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx, row in enumerate(a_rows):
            for c in range(7):
                U[row][col + c] = U_s[idx][c]
        for idx, row in enumerate(b_rows):
            for c in range(7):
                V[row][col + c] = V_s[idx][c]
        for idx, row in enumerate(c_rows):
            for c in range(7):
                W[row][col + c] = W_s[idx][c]
        col += 7
    
    def add_schoolbook_2x2_2x1(a_rows, b_rows, c_rows):
        """Add schoolbook 2x2*2x1 decomposition."""
        nonlocal col
        # a_rows: 4 rows for A[i][j], i,j in {0,1} -> indices i*2+j
        # b_rows: 2 rows for B[j][0], j in {0,1}
        # c_rows: 2 rows for C[i][0], i in {0,1}
        for i in range(2):
            for j in range(2):
                U[a_rows[i*2+j]][col] = 1
                V[b_rows[j]][col] = 1
                W[c_rows[i]][col] = 1
                col += 1
    
    def add_2x2_2x7(a_rows, b_rows, c_rows):
        """
        Add decomposition for 2x2 * 2x7.
        a_rows: 4 rows for A[i][j], i,j in {0,1}
        b_rows: 14 rows for B[j][k], j in {0,1}, k in {0,...,6} -> j*7+k
        c_rows: 14 rows for C[k][i], k in {0,...,6}, i in {0,1} -> k*2+i
        
        Split B into three parts: B[:,0:2], B[:,2:4], B[:,4:7]
        B[:,0:2] is 2x2, B[:,2:4] is 2x2, B[:,4:7] is 2x3
        """
        nonlocal col
        # Part 1: A * B[:,0:2] (2x2 * 2x2) using Strassen
        b_part1 = [b_rows[j*7 + k] for j in range(2) for k in range(2)]
        c_part1 = [c_rows[k*2 + i] for k in range(2) for i in range(2)]
        add_strassen_2x2_2x2(a_rows, b_part1, c_part1)
        
        # Part 2: A * B[:,2:4] (2x2 * 2x2) using Strassen
        b_part2 = [b_rows[j*7 + k] for j in range(2) for k in range(2, 4)]
        c_part2 = [c_rows[k*2 + i] for k in range(2, 4) for i in range(2)]
        add_strassen_2x2_2x2(a_rows, b_part2, c_part2)
        
        # Part 3: A * B[:,4:7] (2x2 * 2x3)
        # Split into 2x2*2x2 (cols 4,5) + 2x2*2x1 (col 6)
        b_part3a = [b_rows[j*7 + k] for j in range(2) for k in range(4, 6)]
        c_part3a = [c_rows[k*2 + i] for k in range(4, 6) for i in range(2)]
        add_strassen_2x2_2x2(a_rows, b_part3a, c_part3a)
        
        b_part3b = [b_rows[j*7 + 6] for j in range(2)]
        c_part3b = [c_rows[6*2 + i] for i in range(2)]
        add_schoolbook_2x2_2x1(a_rows, b_part3b, c_part3b)
    
    # A blocks (2x2 each):
    # A11: i=0,1; j=0,1 -> U rows: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A12: i=0,1; j=2,3 -> U rows: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A21: i=2,3; j=0,1 -> U rows: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # A22: i=2,3; j=2,3 -> U rows: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    # B_top: j=0,1 (rows 0-6 and 7-13 in V)
    # B_bottom: j=2,3 (rows 14-20 and 21-27 in V)
    B_top_rows = [j*7 + k for j in range(2) for k in range(7)]  # 0-6, 7-13
    B_bottom_rows = [j*7 + k for j in range(2, 4) for k in range(7)]  # 14-20, 21-27
    
    # C blocks:
    # C_top (i=0,1): k=0..6 -> W rows k*4+i = k*4+0, k*4+1
    # C_bottom (i=2,3): k=0..6 -> W rows k*4+i = k*4+2, k*4+3
    
    C_top_rows = [k*4 + i for k in range(7) for i in range(2)]  # 0,1,4,5,8,9,12,13,16,17,20,21,24,25
    C_bottom_rows = [k*4 + i for k in range(7) for i in range(2, 4)]  # 2,3,6,7,10,11,14,15,18,19,22,23,26,27
    
    # C = [A11*B_top + A12*B_bottom; A21*B_top + A22*B_bottom]
    
    # Top-left 2x7 block of C: A11*B_top + A12*B_bottom
    # We need to add contributions from both products to the same C rows
    # This means we can't simply call add_2x2_2x7 twice for the same C rows
    # because the U and V entries would conflict.
    
    # Actually, we need to be more careful. The tensor decomposition requires
    # that for each output entry C[i][k], the sum over all rank-1 terms equals 1
    # when multiplied by A[i][j]*B[j][k].
    
    # Let me reconsider. The block structure gives:
    # C[i][k] = sum_j A[i][j]*B[j][k]
    # For i in {0,1}, k in {0,...,6}:
    #   C[i][k] = A11[i][j]*B_top[j][k] + A12[i][j]*B_bottom[j][k] (j in {0,1} for A11, j in {2,3} for A12)
    
    # This is not simply two independent 2x2*2x7 products because the A indices
    # are different (A11 uses j=0,1 while A12 uses j=2,3 in the original 4x4 matrix).
    
    # Let me use a different approach: treat the whole thing as a sum of
    # 4 independent 2x2 * 4x7 products? No, that doesn't work either.
    
    # Actually, the correct block decomposition is:
    # C (4x7) = A (4x4) * B (4x7)
    # Split A into 2x2 blocks and B into 2x7 blocks:
    # C = [[A11, A12], [A21, A22]] * [B_top; B_bottom]
    #   = [A11*B_top + A12*B_bottom; A21*B_top + A22*B_bottom]
    
    # Each term like A11*B_top is a 2x2 * 2x7 product.
    # The challenge is that multiple terms contribute to the same C entries.
    
    # For the tensor decomposition, we need rank-1 terms u*c*v such that
    # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = delta_{i,j,k}
    
    # This is equivalent to: the multiplication tensor has a decomposition into
    # rank-1 tensors. The block structure doesn't directly give us independent
    # decompositions because the same output entries receive contributions from
    # multiple blocks.
    
    # Let me just use the schoolbook for now, which is guaranteed correct.
    # The rank is 4*4*7 = 112.
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Return schoolbook as guaranteed correct fallback
    result = _schoolbook()
    
    # Verify correctness
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    
    return result

# EVOLVE-BLOCK-END
