"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use a structured decomposition that combines Strassen's algorithm for 2x2 blocks
  with schoolbook multiplication for the remaining parts.
- The key idea is to partition the 3x7 matrix B into a 2x7 top part and a 1x7 bottom part,
  and the 2x3 matrix A into a 2x2 left part and a 2x1 right part.
- For the 2x2 * 2x7 part, we can use Strassen's algorithm for each pair of columns,
  reducing the rank from 28 (schoolbook) to 21 (Strassen).
- For the 2x1 * 1x7 part, we use schoolbook (rank 14).
- Total rank: 21 + 14 = 35.

Actually, let me reconsider. The structure is:
A is 2x3, B is 3x7.
C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]

We can write this as:
C = A_left * B_top + A_right * B_bottom

where:
- A_left is 2x2 (columns 0,1 of A)
- B_top is 2x7 (rows 0,1 of B)
- A_right is 2x1 (column 2 of A)
- B_bottom is 1x7 (row 2 of B)

For A_left * B_top (2x2 * 2x7):
We can process this as 7 independent 2x2 * 2x1 multiplications, but that's not efficient.
Better: group the 7 columns into 3 pairs + 1 single.
- Columns (0,1): 2x2 * 2x2 -> Strassen rank 7
- Columns (2,3): 2x2 * 2x2 -> Strassen rank 7
- Columns (4,5): 2x2 * 2x2 -> Strassen rank 7
- Column 6: 2x2 * 2x1 -> schoolbook rank 4
Total for this part: 7+7+7+4 = 25

For A_right * B_bottom (2x1 * 1x7):
This is 2x1 * 1x7, which is just 2*7 = 14 scalar multiplications.
Rank: 14

Total rank: 25 + 14 = 39

Wait, I can do better. Let me think about the 2x2 * 2x1 case.
Actually, for 2x2 * 2x1, the minimal rank is 4 (schoolbook), since it's just 4 independent products.

So total rank is 39.

But wait, can I share structure between the Strassen blocks? No, because they operate on different
columns of B and different entries of C, so they are independent.

Let me verify: 39 < 42, so this is an improvement.

Score: 42/39 ≈ 1.077
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy. Returns True if likely correct."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2(U, V, W, col_offset, a_row_offset, b_row_offset, c_row_offset, n, m, p):
    """
    Add a 2x2 * 2x2 Strassen block.
    
    a_row_offset: row offset in U for the 2x2 A block (rows a_row_offset, a_row_offset+1)
    b_row_offset: row offset in V for the 2x2 B block (rows b_row_offset, b_row_offset+1)
    c_row_offset: row offset in W for the 2x2 C block (rows c_row_offset, c_row_offset+1)
    
    The 2x2 A block has entries at:
    U[a_row_offset][c]     = A[0,0]
    U[a_row_offset+1][c]   = A[0,1]  (wait, this depends on indexing)
    
    Actually, let me be more careful with the indexing.
    
    U has shape (n*m, rank). Row index is i*m + j where i is row of A, j is col of A.
    For A_left (2x2), the rows are:
    - A[0,0]: U[0*3+0] = U[0]
    - A[0,1]: U[0*3+1] = U[1]
    - A[1,0]: U[1*3+0] = U[3]
    - A[1,1]: U[1*3+1] = U[4]
    
    V has shape (m*p, rank). Row index is j*p + k where j is row of B, k is col of B.
    For B_top (2x7), the rows for columns (c0, c1) are:
    - B[0,c0]: V[0*7+c0]
    - B[0,c1]: V[0*7+c1]
    - B[1,c0]: V[1*7+c0]
    - B[1,c1]: V[1*7+c1]
    
    W has shape (p*n, rank). Row index is k*n + i where k is col of C, i is row of C.
    For C (2x7), the rows for columns (c0, c1) are:
    - C[0,c0]: W[c0*2+0]
    - C[0,c1]: W[c1*2+0]
    - C[1,c0]: W[c0*2+1]
    - C[1,c1]: W[c1*2+1]
    
    Strassen's 7 multiplications for 2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 3, 7  # fixed for this problem
    
    # A indices in U: A[i,j] -> U[i*m + j]
    A00 = 0 * m + 0  # U[0]
    A01 = 0 * m + 1  # U[1]
    A10 = 1 * m + 0  # U[3]
    A11 = 1 * m + 1  # U[4]
    
    # B indices in V: B[j,k] -> V[j*p + k]
    # b_row_offset gives the column indices in B
    b00 = 0 * p + b_row_offset    # V[b_row_offset]
    b01 = 0 * p + (b_row_offset + 1)  # V[b_row_offset + 1]
    b10 = 1 * p + b_row_offset    # V[7 + b_row_offset]
    b11 = 1 * p + (b_row_offset + 1)  # V[7 + b_row_offset + 1]
    
    # C indices in W: C[i,k] -> W[k*n + i]
    # c_row_offset gives the column indices in C
    C00 = b_row_offset * n + 0       # W[b_row_offset * 2]
    C01 = (b_row_offset + 1) * n + 0  # W[(b_row_offset + 1) * 2]
    C10 = b_row_offset * n + 1       # W[b_row_offset * 2 + 1]
    C11 = (b_row_offset + 1) * n + 1  # W[(b_row_offset + 1) * 2 + 1]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[A00][col_offset] += 1
    U[A11][col_offset] += 1
    V[b00][col_offset] += 1
    V[b11][col_offset] += 1
    W[C00][col_offset] += 1
    W[C11][col_offset] += 1
    
    # M2 = (A10 + A11)B00
    U[A10][col_offset + 1] += 1
    U[A11][col_offset + 1] += 1
    V[b00][col_offset + 1] += 1
    W[C10][col_offset + 1] += 1
    W[C11][col_offset + 1] -= 1
    
    # M3 = A00(B01 - B11)
    U[A00][col_offset + 2] += 1
    V[b01][col_offset + 2] += 1
    V[b11][col_offset + 2] -= 1
    W[C01][col_offset + 2] += 1
    W[C11][col_offset + 2] += 1
    
    # M4 = A11(B10 - B00)
    U[A11][col_offset + 3] += 1
    V[b10][col_offset + 3] += 1
    V[b00][col_offset + 3] -= 1
    W[C00][col_offset + 3] += 1
    W[C10][col_offset + 3] += 1
    
    # M5 = (A00 + A01)B11
    U[A00][col_offset + 4] += 1
    U[A01][col_offset + 4] += 1
    V[b11][col_offset + 4] += 1
    W[C00][col_offset + 4] -= 1
    W[C01][col_offset + 4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[A10][col_offset + 5] += 1
    U[A00][col_offset + 5] -= 1
    V[b00][col_offset + 5] += 1
    V[b01][col_offset + 5] += 1
    W[C11][col_offset + 5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[A01][col_offset + 6] += 1
    U[A11][col_offset + 6] -= 1
    V[b10][col_offset + 6] += 1
    V[b11][col_offset + 6] += 1
    W[C00][col_offset + 6] += 1
    W[C11][col_offset + 6] += 1
    
    return col_offset + 7

def _schoolbook_2x1_1x7(U, V, W, col_offset, a_col, b_row):
    """
    Add a 2x1 * 1x7 schoolbook block.
    
    a_col: column index in A (0, 1, or 2)
    b_row: row index in B (0, 1, or 2)
    
    This computes C[:, k] += A[:, a_col] * B[b_row, :] for all k.
    """
    n, m, p = 2, 3, 7
    
    # A indices: A[i, a_col] -> U[i*m + a_col]
    # B indices: B[b_row, k] -> V[b_row*p + k]
    # C indices: C[i, k] -> W[k*n + i]
    
    for i in range(n):
        for k in range(p):
            U[i * m + a_col][col_offset] = 1
            V[b_row * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    
    return col_offset

def _strassen_2x2_2x1(U, V, W, col_offset, b_col):
    """
    Add a 2x2 * 2x1 schoolbook block (4 multiplications).
    
    b_col: column index in B
    """
    n, m, p = 2, 3, 7
    
    A00 = 0 * m + 0
    A01 = 0 * m + 1
    A10 = 1 * m + 0
    A11 = 1 * m + 1
    
    b0 = 0 * p + b_col
    b1 = 1 * p + b_col
    
    C00 = b_col * n + 0
    C10 = b_col * n + 1
    
    # A00 * B0
    U[A00][col_offset] = 1
    V[b0][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A01 * B1
    U[A01][col_offset] = 1
    V[b1][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A10 * B0
    U[A10][col_offset] = 1
    V[b0][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    # A11 * B1
    U[A11][col_offset] = 1
    V[b1][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    return col_offset

def _construct_rank_39():
    """
    Construct the rank-39 decomposition.
    
    Part 1: A_left (2x2) * B_top (2x7)
    - Columns (0,1): Strassen 2x2*2x2, rank 7
    - Columns (2,3): Strassen 2x2*2x2, rank 7
    - Columns (4,5): Strassen 2x2*2x2, rank 7
    - Column 6: Schoolbook 2x2*2x1, rank 4
    Total: 25
    
    Part 2: A_right (2x1) * B_bottom (1x7)
    - Schoolbook 2x1*1x7, rank 14
    Total: 14
    
    Grand total: 39
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    rank = 39
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Part 1: A_left * B_top
    # Strassen for columns (0,1)
    col = _strassen_2x2(U, V, W, col, 0, 0, 0, n, m, p)
    
    # Strassen for columns (2,3)
    col = _strassen_2x2(U, V, W, col, 0, 2, 2, n, m, p)
    
    # Strassen for columns (4,5)
    col = _strassen_2x2(U, V, W, col, 0, 4, 4, n, m, p)
    
    # Schoolbook for column 6
    col = _strassen_2x2_2x1(U, V, W, col, 6)
    
    # Part 2: A_right * B_bottom
    col = _schoolbook_2x1_1x7(U, V, W, col, 2, 2)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try the rank-39 construction first
    result = _construct_rank_39()
    
    # Verify it's correct
    if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
        if _verify_decomposition_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
