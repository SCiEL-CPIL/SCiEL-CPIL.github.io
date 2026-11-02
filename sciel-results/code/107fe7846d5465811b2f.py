"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _construct_strassen_2x3x3():
    """
    Construct a rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    
    Strategy: Split the 3x3 matrix into blocks:
    - Left 2x2 block (columns 0,1): use Strassen (rank 7) for the 2x2*2x2 part
    - Right 1x3 block (column 2): use schoolbook (rank 6) for 2x1*1*3
    
    Wait, that's not quite right. Let me think more carefully.
    
    We want to compute C = A * B where A is 2x3 and B is 3x3.
    
    Split A into [A1 A2] where A1 is 2x2 (first two columns) and A2 is 2x1 (third column).
    Split B into [B1 B2] where B1 is 2x3 (first two columns) and B2 is 1x3 (third column).
    
    Then C = A1*B1 + A2*B2.
    
    A1*B1 is 2x2 * 2x3. We can compute this using Strassen-like methods.
    Actually, let's think of it differently.
    
    Let me use a different decomposition. Split B by rows:
    B = [B_top; B_bottom] where B_top is 2x3 and B_bottom is 1x3.
    A = [A1 A2 A3] where A1, A2 are 2x1 and A3 is 2x1.
    
    C = A1 * B_top_row1 + A2 * B_top_row2 + A3 * B_bottom_row
    
    Hmm, this is getting complicated. Let me use a known approach.
    
    Actually, the known optimal rank for 2x3x3 is 17, achieved by:
    - Using Strassen's algorithm on a 2x2x2 subproblem (rank 7)
    - Using schoolbook on the remaining parts
    
    Let me construct it explicitly.
    
    We have A (2x3) times B (3x3) = C (2x3).
    
    Split columns of A: A = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
    Split rows of B: B = [B_top; B_bottom] where B_top is 2x3, B_bottom is 1x3.
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 * 2x3):
    We can split B_top into columns: B_top = [B_t1 | B_t2 | B_t3] where each is 2x1.
    So A_left * B_top = [A_left*B_t1 | A_left*B_t2 | A_left*B_t3]
    Each A_left*B_tj is 2x2 * 2x1, which has rank 4 (schoolbook) or can be done better.
    Actually 2x2*2x1: the tensor is 2x2x1, rank is at most 4.
    
    This is getting too complex. Let me just use a direct construction.
    
    Known result: rank(2x3x3) = 17.
    
    Let me use the following approach:
    - 7 terms from Strassen for a 2x2x2 core
    - 10 terms from schoolbook for the rest
    
    Actually, let me just verify the rank-17 construction directly.
    
    I'll use a simpler approach: construct the decomposition and verify it.
    
    The key insight is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i in {0,1}, k in {0,1,2}, j in {0,1,2}.
    
    Total 18 scalar products in schoolbook.
    
    To get rank 17, we need to combine one pair of terms.
    
    Let's use the identity: for the 2x2 part (i in {0,1}, j in {0,1}, k in {0,1}),
    we can use Strassen's 7 multiplications instead of 4.
    
    Strassen for 2x2*2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Now for the full 2x3*3x3:
    A = [[A00, A01, A02], [A10, A11, A12]]
    B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    
    C = A * B where C is 2x3.
    
    C[0,0] = A00*B00 + A01*B10 + A02*B20
    C[0,1] = A00*B01 + A01*B11 + A02*B21
    C[0,2] = A00*B02 + A01*B12 + A02*B22
    C[1,0] = A10*B00 + A11*B10 + A12*B20
    C[1,1] = A10*B01 + A11*B11 + A12*B21
    C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    Using Strassen for the 2x2 part (rows 0,1 of A; rows 0,1 of B; cols 0,1 of B):
    The 2x2 subproblem is A_sub * B_sub where:
    A_sub = [[A00, A01], [A10, A11]] (2x2)
    B_sub = [[B00, B01], [B10, B11]] (2x2)
    
    This gives C_sub (2x2) = A_sub * B_sub, contributing to C[0,0], C[0,1], C[1,0], C[1,1].
    
    The remaining terms involve A02, A12 (third column of A) and B20, B21, B22 (third row of B),
    and also B02, B12 (third column of B, first two rows).
    
    Additional terms needed:
    C[0,0] += A02*B20
    C[0,1] += A02*B21
    C[0,2] = A00*B02 + A01*B12 + A02*B22
    C[1,0] += A12*B20
    C[1,1] += A12*B21
    C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    So we need:
    - 7 Strassen terms for the 2x2 core
    - A02*B20 (1 term)
    - A02*B21 (1 term)
    - A12*B20 (1 term)
    - A12*B21 (1 term)
    - A00*B02 (1 term)
    - A01*B12 (1 term)
    - A02*B22 (1 term)
    - A10*B02 (1 term)
    - A11*B12 (1 term)
    - A12*B22 (1 term)
    
    That's 7 + 10 = 17 terms. 
    
    Now let me construct U, V, W matrices.
    
    U has shape (2*3, 17) = (6, 17), indexed by [i*m+j] = [i*3+j]
    V has shape (3*3, 17) = (9, 17), indexed by [j*p+k] = [j*3+k]
    W has shape (3*2, 17) = (6, 17), indexed by [k*n+i] = [k*2+i]
    
    Strassen terms (r=0..6):
    r=0: M1 = (A00+A11)*(B00+B11)
      U[0,0]=1, U[4,0]=1 (A00, A11)
      V[0,0]=1, V[4,0]=1 (B00, B11)
      W[0,0]=1, W[1,0]=1, W[2,0]=-1, W[3,0]=1 (C00+=M1, C01+=M1, C10+=M1, C11-=M1)
      Wait, let me re-check Strassen:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
      
      So M1 contributes: C00+=1, C01+=0, C10+=0, C11+=1
      
    Let me redo this carefully.
    
    r=0: M1 = (A00+A11)*(B00+B11)
      U: A00 (idx 0) += 1, A11 (idx 4) += 1
      V: B00 (idx 0) += 1, B11 (idx 4) += 1
      W: C00 (idx 0) += 1, C11 (idx 3) += 1
    
    r=1: M2 = (A10+A11)*B00
      U: A10 (idx 3) += 1, A11 (idx 4) += 1
      V: B00 (idx 0) += 1
      W: C10 (idx 2) += 1, C11 (idx 3) -= 1
    
    r=2: M3 = A00*(B01-B11)
      U: A00 (idx 0) += 1
      V: B01 (idx 1) += 1, B11 (idx 4) -= 1
      W: C01 (idx 1) += 1, C11 (idx 3) += 1
    
    r=3: M4 = A11*(B10-B00)
      U: A11 (idx 4) += 1
      V: B10 (idx 3) += 1, B00 (idx 0) -= 1
      W: C00 (idx 0) += 1, C10 (idx 2) += 1
    
    r=4: M5 = (A00+A01)*B11
      U: A00 (idx 0) += 1, A01 (idx 1) += 1
      V: B11 (idx 4) += 1
      W: C00 (idx 0) -= 1, C01 (idx 1) += 1
    
    r=5: M6 = (A10-A00)*(B00+B01)
      U: A10 (idx 3) += 1, A00 (idx 0) -= 1
      V: B00 (idx 0) += 1, B01 (idx 1) += 1
      W: C11 (idx 3) += 1
    
    r=6: M7 = (A01-A11)*(B10+B11)
      U: A01 (idx 1) += 1, A11 (idx 4) -= 1
      V: B10 (idx 3) += 1, B11 (idx 4) += 1
      W: C00 (idx 0) += 1
    
    Schoolbook terms (r=7..16):
    r=7: A02*B20
      U: A02 (idx 2) += 1
      V: B20 (idx 6) += 1
      W: C00 (idx 0) += 1
    
    r=8: A02*B21
      U: A02 (idx 2) += 1
      V: B21 (idx 7) += 1
      W: C01 (idx 1) += 1
    
    r=9: A12*B20
      U: A12 (idx 5) += 1
      V: B20 (idx 6) += 1
      W: C10 (idx 2) += 1
    
    r=10: A12*B21
      U: A12 (idx 5) += 1
      V: B21 (idx 7) += 1
      W: C11 (idx 3) += 1
    
    r=11: A00*B02
      U: A00 (idx 0) += 1
      V: B02 (idx 2) += 1
      W: C02 (idx 4) += 1
    
    r=12: A01*B12
      U: A01 (idx 1) += 1
      V: B12 (idx 5) += 1
      W: C02 (idx 4) += 1
    
    r=13: A02*B22
      U: A02 (idx 2) += 1
      V: B22 (idx 8) += 1
      W: C02 (idx 4) += 1
    
    r=14: A10*B02
      U: A10 (idx 3) += 1
      V: B02 (idx 2) += 1
      W: C12 (idx 5) += 1
    
    r=15: A11*B12
      U: A11 (idx 4) += 1
      V: B12 (idx 5) += 1
      W: C12 (idx 5) += 1
    
    r=16: A12*B22
      U: A12 (idx 5) += 1
      V: B22 (idx 8) += 1
      W: C12 (idx 5) += 1
    
    Let me now construct the matrices and verify.
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 17)
    V = [[0] * rank for _ in range(m * p)]  # (9, 17)
    W = [[0] * rank for _ in range(p * n)]  # (6, 17)
    
    # Strassen terms
    # r=0: M1 = (A00+A11)*(B00+B11)
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # r=1: M2 = (A10+A11)*B00
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # r=2: M3 = A00*(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # r=3: M4 = A11*(B10-B00)
    U[4][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # r=4: M5 = (A00+A01)*B11
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # r=5: M6 = (A10-A00)*(B00+B01)
    U[3][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # r=6: M7 = (A01-A11)*(B10+B11)
    U[1][6] += 1; U[4][6] -= 1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Schoolbook terms
    # r=7: A02*B20
    U[2][7] += 1
    V[6][7] += 1
    W[0][7] += 1
    
    # r=8: A02*B21
    U[2][8] += 1
    V[7][8] += 1
    W[1][8] += 1
    
    # r=9: A12*B20
    U[5][9] += 1
    V[6][9] += 1
    W[2][9] += 1
    
    # r=10: A12*B21
    U[5][10] += 1
    V[7][10] += 1
    W[3][10] += 1
    
    # r=11: A00*B02
    U[0][11] += 1
    V[2][11] += 1
    W[4][11] += 1
    
    # r=12: A01*B12
    U[1][12] += 1
    V[5][12] += 1
    W[4][12] += 1
    
    # r=13: A02*B22
    U[2][13] += 1
    V[8][13] += 1
    W[4][13] += 1
    
    # r=14: A10*B02
    U[3][14] += 1
    V[2][14] += 1
    W[5][14] += 1
    
    # r=15: A11*B12
    U[4][15] += 1
    V[5][15] += 1
    W[5][15] += 1
    
    # r=16: A12*B22
    U[5][16] += 1
    V[8][16] += 1
    W[5][16] += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    # If verification fails, fall back to schoolbook
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # First try the rank-17 Strassen-based decomposition
    result = _construct_strassen_2x3x3()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
