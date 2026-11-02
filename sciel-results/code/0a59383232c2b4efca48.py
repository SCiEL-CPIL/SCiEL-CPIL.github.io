"""Matrix multiplication 2x3x9: rank-minimal exact tensor decomposition.

Strategy:
  We treat the 2x3 x 3x9 multiplication as follows:
  - The 2x3x3 multiplication tensor has rank 6 (Strassen's algorithm).
  - The 3x9 multiplication can be seen as 3 independent 1x3 multiplications
    (since the left matrix has 2 rows, each row is a 1x3 vector, and the right
    matrix is 3x9, so each of the 2 rows produces a 1x9 result via 3 scalar
    multiplications summed over the 3 columns).
  
  Actually, let me think more carefully.
  
  For a 2x3 matrix A and 3x9 matrix B, the product C = AB is 2x9.
  C[i][k] = sum_{j=0}^{2} A[i][j] * B[j][k]
  
  So we have 2*3*9 = 54 terms total.
  
  The schoolbook rank is 54. Can we do better?
  
  Let's think about this using algebraic approaches.
  
  Consider the structure: we're multiplying a 2x3 matrix by a 3x9 matrix.
  This is equivalent to 2 independent 1x3 by 3x9 multiplications.
  
  A 1x3 by 3x9 multiplication: C[k] = sum_{j=0}^{2} a[j] * B[j][k] for k=0..8
  This is just 9 scalar multiplications... but wait, they share the same a[j].
  
  Actually, for a single 1x3 by 3x9: the tensor has dimensions (3, 9) on the input
  side (a is 3-dim, B is 27-dim) and output is 9-dim. The rank of this tensor is
  at most 27 (schoolbook). But since a is shared across all 9 outputs, we might
  be able to do better.
  
  Hmm, let me think differently. 
  
  Actually, the tensor for matrix multiplication A(m,n) x B(n,p) -> C(m,p) has:
  - U dimension: m*n
  - V dimension: n*p  
  - W dimension: p*m
  - Rank: at most m*n*p (schoolbook)
  
  For 2x3 x 3x9: m=2, n=3, p=9
  - U dim: 6
  - V dim: 27
  - W dim: 18
  - Schoolbook rank: 54
  
  Known lower bounds: rank >= max(mn, np, pm) = max(6, 27, 18) = 27
  
  Can we achieve rank 27? That would require each term to contribute to multiple
  products efficiently.
  
  Let me try a different approach. Since p=9 = 3^2, maybe there's a structure.
  
  Actually, let me just implement the schoolbook and see if I can find a better
  decomposition through analysis.
  
  For now, let me implement the schoolbook decomposition (rank 54) as a baseline,
  and then try to improve it.
"""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 x 3x9 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (6, rank)
        - V has shape (m*p, rank) = (27, rank)
        - W has shape (p*n, rank) = (18, rank)
    
    The decomposition satisfies: for all i,k, C[i,k] = sum_r U[i*m + ? , r] * V[?, r] * W[k*n + i, r]
    Wait, let me check the convention: W[k*n+i, r] where n=2, m=3, p=9.
    
    Actually, looking at the schoolbook code:
    - U[i*m + j, c] where i is row of A (0..1), j is col of A / row of B (0..2)
    - V[j*p + k, c] where j is row of B (0..2), k is col of B (0..8)
    - W[k*n + i, c] where k is col of C (0..8), i is row of C (0..1)
    
    So U has indices (i, j) -> i*3 + j, shape (6, rank)
    V has indices (j, k) -> j*9 + k, shape (27, rank)
    W has indices (k, i) -> k*2 + i, shape (18, rank)
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = n * m * p  # 54 for schoolbook
    U = [[0] * rank for _ in range(n * m)]   # 6 x rank
    V = [[0] * rank for _ in range(m * p)]   # 27 x rank
    W = [[0] * rank for _ in range(p * n)]   # 18 x rank
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
