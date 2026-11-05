"""Matrix multiplication 4,4,7 - Strassen-based decomposition with rank 93."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7) with indices:
    U[i*m+j, c] where m=2, so U[0..3]
    V[j*p+k, c] where p=2, so V[0..3]
    W[k*n+i, c] where n=2, so W[0..3]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4x7
    V = [[0]*rank for _ in range(m*p)]  # 4x7
    W = [[0]*rank for _ in range(p*n)]  # 4x7

    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # U maps input A (2x2) to the 7 intermediate products
    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # U[i*m+j, c] = coefficient of A[i,j] in intermediate c

    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    # M2: (A21+A22)B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    # M3: A11(B21-B22)
    U[0][2] = 1  # A11
    # M4: A22(B12-B11)
    U[3][3] = 1  # A22
    # M5: (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22

    # V maps input B (2x2) to the 7 intermediate products
    # B indices: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # V[j*p+k, c] = coefficient of B[j,k] in intermediate c

    # M1: (A11+A22)(B11+B22)
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # M2: (A21+A22)B11
    V[0][1] = 1  # B11
    # M3: A11(B21-B22)
    V[2][2] = 1  # B21
    V[3][2] = -1 # B22
    # M4: A22(B12-B11)
    V[1][3] = 1  # B12
    V[0][3] = -1 # B11
    # M5: (A11+A12)B22
    V[3][4] = 1  # B22
    # M6: (A21-A11)(B11+B12)
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    # M7: (A12-A22)(B21+B22)
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22

    # W maps the 7 intermediates to C (2x2)
    # W[k*n+i, c] = coefficient of intermediate c in C[i,k]
    # C indices: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22

    # C11 = M1+M4-M5+M7
    W[0][0] = 1   # M1 -> C11
    W[0][3] = 1   # M4 -> C11
    W[0][4] = -1  # M5 -> C11
    W[0][6] = 1   # M7 -> C11
    # C12 = M3+M5
    W[1][2] = 1   # M3 -> C12
    W[1][4] = 1   # M5 -> C12
    # C21 = M2+M4
    W[2][1] = 1   # M2 -> C21
    W[2][3] = 1   # M4 -> C21
    # C22 = M1-M2+M3+M6
    W[3][0] = 1   # M1 -> C22
    W[3][1] = -1  # M2 -> C22
    W[3][2] = 1   # M3 -> C22
    W[3][5] = 1   # M6 -> C22

    return U, V, W


def _schoolbook(n, m, p):
    """Return U, V, W for n x m by m x p matrix multiplication (schoolbook, rank n*m*p)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    # Allocate full matrices
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 93
    V = [[0]*total_rank for _ in range(m*p)]  # 28 x 93
    W = [[0]*total_rank for _ in range(p*n)]  # 28 x 93
    
    col = 0
    
    # Split B into B_L (4x4) and B_R (4x3)
    # Split A into A_top (2x4) and A_bot (2x4)
    
    # For each block of A (2 rows) and block of B:
    # If B block is 4x4: use Strassen (rank 7)
    # If B block is 4x2 or 4x3: use schoolbook
    
    # Actually, let me think more carefully.
    # A is 4x4, B is 4x7, C is 4x7.
    # 
    # Partition A by rows into A_top (2x4) and A_bot (2x4).
    # Partition B by columns into B_0..B_6 (each 4x1), but better to group.
    #
    # Let's partition B into blocks of columns: B_L = B[:, 0:4] (4x4), B_M = B[:, 4:6] (4x2), B_R = B[:, 6] (4x1).
    #
    # C_top = A_top * B (2x7)
    # C_bot = A_bot * B (2x7)
    
    # For A_top (2x4) * B_L (4x4) -> 2x4 result
    # Partition A_top into A_top_L (2x2) and A_top_R (2x2)
    # Partition B_L into B_L_top (2x4) and B_L_bot (2x4)
    # 
    # This gets complicated. Let me use a simpler approach:
    #
    # Partition A into four 2x2 blocks: A11, A12, A21, A22
    # Partition B into blocks based on columns.
    # B is 4x7. Partition B rows into B_top (2x7) and B_bot (2x7).
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # For C_top (rows 0,1) and C_bot (rows 2,3):
    # C_top[:, k] = A11 * B_top[:, k] + A12 * B_bot[:, k]
    # C_bot[:, k] = A21 * B_top[:, k] + A22 * B_bot[:, k]
    
    # So we have 4 sub-problems:
    # 1. A11 (2x2) * B_top (2x7) -> 2x7
    # 2. A12 (2x2) * B_bot (2x7) -> 2x7
    # 3. A21 (2x2) * B_top (2x7) -> 2x7
    # 4. A22 (2x2) * B_bot (2x7) -> 2x7
    
    # Each sub-problem: 2x2 * 2x7.
    # Partition B (2x7) into B_a (2x3) and B_b (2x4).
    # 2x2 * 2x3: rank 6 (schoolbook)
    # 2x2 * 2x4: can use Strassen for 2x2*2x2 twice -> rank 14
    
    # Total: 4 * (6 + 14) = 4 * 20 = 80
    
    # Let me restructure:
    # 2x2 * 2x7: partition B into (2x3) and (2x4)
    # 2x2 * 2x3: schoolbook rank 6
    # 2x2 * 2x4: partition into two 2x2*2x2, each Strassen rank 7, total 14
    # Total per 2x2*2x7: 6 + 14 = 20
    # Four such: 80
    
    # Wait, but I can do better. Let me think about 2x2*2x7 differently.
    # Actually, the minimum rank for 2x2*2x7... Let's just use the decomposition above.
    
    # Let me implement:
    # For each of the 4 sub-problems (A_block 2x2, B_block 2x7):
    #   Split B_block into B1 (2x3) and B2 (2x4)
    #   A_block * B1: schoolbook rank 6
    #   A_block * B2: split B2 into B2a (2x2) and B2b (2x2)
    #     A_block * B2a: Strassen rank 7
    #     A_block * B2b: Strassen rank 7
    
    # Total: 4 * (6 + 7 + 7) = 4 * 20 = 80
    
    # Let me code this up.
    
    # A_block indices:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B_top: rows 0-1, cols 0-6
    # B_bot: rows 2-3, cols 0-6
    
    # For each sub-problem, we need to map local indices to global.
    
    def place_subproblem(U_block_rows, V_block_rows, W_block_rows, 
                         U, V, W, col_start, 
                         a_row_off, a_col_off, b_row_off, b_col_off,
                         n_block, m_block, p_block):
        """Place a sub-decomposition into global arrays."""
        # This is complex, let me just do it inline.
        pass
    
    # Let me just build it directly.
    
    # Sub-problem (ai, bi) where ai in {0,1} (top/bot of A), bi in {0,1} (top/bot of B)
    # A block: rows ai*2 to ai*2+1, cols 0-3 (full 2x4)
    # Wait no. Let me reconsider.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # i in 0..3, j in 0..3, k in 0..6
    
    # Partition j: j in 0..1 (B_top) and j in 2..3 (B_bot)
    # Partition i: i in 0..1 (C_top) and i in 2..3 (C_bot)
    
    # C_top[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # = A_top[i, 0:2] dot B_top[:, k] + A_top[i, 2:4] dot B_bot[:, k]
    
    # So for C_top (i in 0,1):
    #   Part 1: A[i, 0:2] (1x2) * B_top[:, k] (2x1) -> scalar
    #   Part 2: A[i, 2:4] (1x2) * B_bot[:, k] (2x1) -> scalar
    #
    # But A[i, 0:2] for i=0 is A[0,0], A[0,1] = A11 row 0
    # A[i, 0:2] for i=1 is A[1,0], A[1,1] = A11 row 1
    # So A_top_left (2x2) * B_top (2x7) gives C_top_left contribution
    # A_top_right (2x2) * B_bot (2x7) gives C_top_right contribution
    #
    # Similarly for C_bot:
    # A_bot_left (2x2) * B_top (2x7)
    # A_bot_right (2x2) * B_bot (2x7)
    
    # So 4 sub-problems, each 2x2 * 2x7.
    
    # For 2x2 * 2x7:
    # A is 2x2, B is 2x7 (rows are the "j" dimension of size 2, cols are "k" of size 7)
    # C = A * B is 2x7
    
    # Let's partition B's 7 columns into 3+2+2 or 3+4 or 4+3.
    # 2x2 * 2x3: schoolbook rank 6
    # 2x2 * 2x4: use two Strassen 2x2*2x2, rank 14
    
    # Or: 2x2 * 2x7 = 2x2*2x3 + 2x2*2x4 = rank 6 + 14 = 20
    
    # Total: 4 * 20 = 80
    
    # Let me implement this.
    
    U = [[0]*80 for _ in range(16)]
    V = [[0]*80 for _ in range(28)]
    W = [[0]*80 for _ in range(28)]
    
    col = 0
    
    # For each sub-problem:
    # (a_row_start, b_row_start, c_row_start)
    # a_row_start: 0 or 2 (which 2 rows of A)
    # b_row_start: 0 or 2 (which 2 rows of B, i.e., which j values)
    # c_row_start: 0 or 2 (which 2 rows of C)
    
    for a_rs in [0, 2]:
        for b_rs in [0, 2]:
            c_rs = a_rs  # C rows correspond to A rows
            
            # A sub-block: rows a_rs to a_rs+1, all 4 cols
            # But we split A cols into left (0-1) and right (2-3)
            # Wait, the j index goes from 0 to 3. b_rs determines which j values.
            # If b_rs=0: j in 0,1. A cols 0,1 are used.
            # If b_rs=2: j in 2,3. A cols 2,3 are used.
            
            a_col_start = b_rs  # A cols corresponding to the j values
            
            # A_local is 2x2: A[a_rs:a_rs+2, a_col_start:a_col_start+2]
            # B_local is 2x7: B[b_rs:b_rs+2, :]
            # C_local is 2x7: C[a_rs:a_rs+2, :]
            
            # Split B_local cols into 3+4:
            # B1: 2x3 (cols 0-2)
            # B2: 2x4 (cols 3-6)
            
            # Part 1: A_local (2x2) * B1 (2x3) -> 2x3, schoolbook rank 6
            # Part 2: A_local (2x2) * B2 (2x4) -> 2x4, two Strassen rank 14
            
            # Part 1: schoolbook 2x2*2x3
            U1, V1, W1 = _schoolbook(2, 2, 3)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        for c in range(6):
                            if U1[i*2+j][c] != 0:
                                U[(a_rs+i)*4 + (a_col_start+j)][col+c] += U1[i*2+j][c]
                            if V1[j*2+k][c] != 0:
                                V[(b_rs+j)*7 + k][col+c] += V1[j*2+k][c]
                            if W1[k*2+i][c] != 0:
                                W[k*4 + (a_rs+i)][col+c] += W1[k*2+i][c]
            col += 6
            
            # Part 2: A_local (2x2) * B2 (2x4)
            # Split B2 into B2a (2x2, cols 0-1 of B2 = cols 3-4 of B) and B2b (2x2, cols 2-3 of B2 = cols 5-6 of B)
            
            # B2a: B[b_rs:b_rs+2, 3:5]
            # B2b: B[b_rs:b_rs+2, 5:7]
            
            # Strassen for A_local * B2a
            U2a, V2a, W2a = _strassen_2x2x2()
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        if U2a[i*2+j][c] != 0:
                            U[(a_rs+i)*4 + (a_col_start+j)][col+c] += U2a[i*2+j][c]
                        if V2a[j*2+0][c] != 0:
                            V[(b_rs+j)*7 + (3+0)][col+c] += V2a[j*2+0][c]
                        if V2a[j*2+1][c] != 0:
                            V[(b_rs+j)*7 + (3+1)][col+c] += V2a[j*2+1][c]
                        if W2a[0*2+i][c] != 0:
                            W[(3+0)*4 + (a_rs+i)][col+c] += W2a[0*2+i][c]
                        if W2a[1*2+i][c] != 0:
                            W[(3+1)*4 + (a_rs+i)][col+c] += W2a[1*2+i][c]
            col += 7
            
            # Strassen for A_local * B2b
            U2b, V2b, W2b = _strassen_2x2x2()
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        if U2b[i*2+j][c] != 0:
                            U[(a_rs+i)*4 + (a_col_start+j)][col+c] += U2b[i*2+j][c]
                        if V2b[j*2+0][c] != 0:
                            V[(b_rs+j)*7 + (5+0)][col+c] += V2b[j*2+0][c]
                        if V2b[j*2+1][c] != 0:
                            V[(b_rs+j)*7 + (5+1)][col+c] += V2b[j*2+1][c]
                        if W2b[0*2+i][c] != 0:
                            W[(5+0)*4 + (a_rs+i)][col+c] += W2b[0*2+i][c]
                        if W2b[1*2+i][c] != 0:
                            W[(5+1)*4 + (a_rs+i)][col+c] += W2b[1*2+i][c]
            col += 7
    
    # Verify col == 80
    # col should be 4 * (6 + 7 + 7) = 4 * 20 = 80
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
