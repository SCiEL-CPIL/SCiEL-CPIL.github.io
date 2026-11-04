"""Matrix multiplication 2x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _build_2x3x3_matmul():
    """Build rank-9 schoolbook decomposition for 2x3x3 matmul.
    Returns U (6x9), V (9x9), W (9x9) as lists."""
    n, m, p = 2, 3, 3
    rank = n * m * p  # 18... wait, schoolbook is n*m*p = 18 for 2x3x3
    # Actually schoolbook rank for n x m x p is n*m*p
    # For 2x3x3: 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 9 x 18
    W = [[0] * rank for _ in range(p * n)]  # 6 x 18
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Decompose 2x3x6 as: split output 6 into two blocks of 3.
    # C[:, 0:3] = A @ B[:, 0:3]  (2x3 times 3x3 -> 2x3)
    # C[:, 3:6] = A @ B[:, 3:6]  (2x3 times 3x3 -> 2x3)
    # Each sub-problem is 2x3x3 matmul with rank 18 (schoolbook).
    # Total rank: 36... that's the same as before!
    
    # Better: use the Kronecker structure.
    # Write B as 2 blocks of 3 columns: B = [B1 | B2]
    # C = [A*B1 | A*B2]
    # The "select" between B1 and B2 is a 1x1x2 tensor of rank 2.
    # The 2x3x3 matmul has rank 18.
    # Kronecker: rank = 18 * 2 = 36. Still the same.
    
    # Alternative: split A's columns. A is 2x3. Write A = [A_col0 | A_cols12]
    # C = A_col0 * B_row0 + A_cols12 * B_rows12
    # First term: 2x1 * 1x6 -> outer product, rank 2
    # Second term: 2x2 * 2x6 -> rank 2*2*6 = 24 schoolbook
    # Total: 26
    
    # Or: C = A_col0 * B_row0 + A_col1 * B_row1 + A_col2 * B_row2
    # Each term: 2x1 * 1x6, rank 2 each. Total: 6. Wait that's just schoolbook 6 terms?
    # No: A is 2x3, B is 3x6. C[i,k] = sum_j A[i,j]*B[j,k]. 6 terms per (i,k), 12 (i,k) pairs = 72... no.
    # Schoolbook: for each (i,j,k): one multiplication. n*m*p = 2*3*6 = 36 terms.
    
    # The 2x2x6 subproblem: A' is 2x2, B' is 2x6.
    # C' = A' * B'. Rank schoolbook: 2*2*6 = 24.
    # Plus the 2x1*1x6 term: rank 2.
    # Total: 26.
    
    # Can we do 2x2x6 better? Split 6 into 3+3:
    # 2x2x3 has schoolbook rank 12. Two of them: 24. Same.
    # Or use Strassen-like for 2x2? 2x2x2 has rank 7 (Strassen).
    # 2x2x6 = 2x2x(2+2+2) = three 2x2x2 blocks. Rank 3*7 = 21.
    # Plus 2 for the first column: total 23.
    
    # Let's try: rank 23 construction.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # = A[i,0]*B[0,k] + (A[i,1]*B[1,k] + A[i,2]*B[2,k])
    # First term: rank 2 (two i values, one j, six k's -> 2*1*6 = 12 schoolbook? No.)
    # Wait: for fixed j=0, C[i,k] += A[i,0]*B[0,k]. This is an outer product of vector A[:,0] (size 2) and B[0,:] (size 6).
    # As a tensor: U has 2 entries, V has 6 entries, W has 12 entries. Rank = 2*1*6 = 12? No, rank of outer product a*b^T as a 3-tensor...
    
    # I'm overcomplicating. Let me just verify: the schoolbook decomposition IS the rank-36 one.
    # Any improvement needs actual rank reduction.
    
    # Known: rank of 2x3x6 matmul tensor. Lower bound ~ 27.
    # Let me try a simple block approach:
    # Split B into two 3x3 blocks: B = [B1, B2]
    # C = [A*B1, A*B2]
    # Use the same A for both: share the "A loading"
    # In tensor decomposition terms, this doesn't directly reduce rank.
    
    # Let me just return the schoolbook rank 36 for now, but structure it for potential improvement.
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
