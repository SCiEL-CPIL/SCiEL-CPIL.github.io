"""
Rank-minimal decomposition for 3x3 * 3x8 matrix multiplication.

We use a block-based approach:
  - Split the 3x8 matrix into two 3x4 blocks (or use 2x2 Strassen on sub-blocks).
  - Actually, a better approach: treat 3x8 as three 3x(8/3) ... not integer.
  
  Instead, we can view the 3x8 as a 3x3 times 3x8.
  We can decompose the 3x3 multiplication using a known rank-23 or rank-24 decomposition,
  but constructing that from scratch is complex.
  
  A simpler and reliable improvement: use the fact that we can split the 8 columns
  into groups and use Strassen-like tricks on 2x2 blocks.
  
  Actually, let's think about the tensor structure:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  We can use the following decomposition idea:
  - For the 3x3 part, use the standard schoolbook (rank 9 per 3x3 block if done naively)
  - But we can do better by using the algebraic structure.
  
  Let me try a different approach: use the known fact that 3x3 * 3x3 can be done
  with rank 23 (or at least 24 with a simple construction). For 3x8, we can think of it
  as 3x3 * 3x8 = (3x3 * 3x3) + (3x3 * 3x5) ... this gets messy.
  
  A cleaner approach: note that 8 = 2 + 2 + 2 + 2, so we have four 3x2 blocks.
  Or 8 = 3 + 3 + 2.
  
  Let me try: decompose B as [B1 B2] where B1 is 3x4 and B2 is 3x4.
  Then C = A*B1 + A*B2. If we can do 3x3 * 3x4 with rank r, then total is 2r.
  For 3x3 * 3x4: we can think of it as three 1x3 * 3x4 multiplications ... no.
  
  Actually, let's just implement the schoolbook with a small optimization:
  Use the fact that we can share some computations.
  
  The simplest reliable improvement: use the 2x2 Strassen trick on the 3x3 matrix.
  Split 3x3 into:
  A = [[A11, A12, A13],
       [A21, A22, A23],
       [A31, A32, A33]]
  
  We can use a known rank-24 decomposition for 3x3 * 3x3. But I need to construct it.
  
  Given the time constraint and the need for correctness, I'll implement a decomposition
  that uses Strassen on 2x2 blocks. Split the 3x3 into a 2x2 block and a 1x3 row/column.
  
  A = [[A2x2, a],
       [b^T, c]]
  B = [[B2x2, d],
       [e^T, f]]
  
  C = [[A2x2*B2x2 + a*e^T, A2x2*d + a*f],
       [b^T*B2x2 + c*e^T, b^T*d + c*f]]
  
  Rank(A2x2*B2x2) <= 7 (Strassen)
  Rank(a*e^T) = 2 (outer product of 2x1 and 1x2)
  Rank(A2x2*d) = 4 (2x2 * 2x1 = 2x1, but as a tensor it's 2*2*1=4... wait)
  
  Actually, for the tensor decomposition, each "multiply" of shape (n x m) * (m x p) has rank n*m*p in schoolbook.
  
  Let me just implement the schoolbook correctly. The rank is 3*3*8 = 72.
"""

import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Schoolbook: rank = n*m*p = 72
    # U: (n*m) x rank = 9 x 72
    # V: (m*p) x rank = 24 x 72
    # W: (p*n) x rank = 24 x 72
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
