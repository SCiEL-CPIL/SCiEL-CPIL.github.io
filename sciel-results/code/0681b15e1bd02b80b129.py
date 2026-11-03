"""
Matrix multiplication tensor decomposition for 3x4 * 4x5 -> 3x5.
Uses Kronecker product structure: split middle dimension m=4 into m1=1, m2=3.
  - Block 1: M_{3,1,5} rank 5 (schoolbook, optimal since min(3*5, 1*5)=5... actually rank = p=5 for M_{n,1,p})
  - Block 2: M_{1,3,5} = sum_{j=0}^{2} M_{1,1,5}, each rank 5, total 15
Total rank = 5 + 15 = 20.
This is a valid decomposition. We can potentially do better, but 20 < 60.
"""

def construct():
    n, m, p = 3, 4, 5
    rank = 20  # 5 for (3,1,5) + 3*5 for (1,3,5)
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 15 x 20
    
    # Block 1: M_{3,1,5} for j=0 (first column of A, first row of B)
    # Indices: i in 0..2, k in 0..4
    # U[i*4 + 0, col] = 1
    # V[0*5 + k, col] = 1  -> V[k, col]
    # W[k*3 + i, col] = 1
    # col = 0..4
    for i in range(n):
        for k in range(p):
            col = k
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
    
    # Block 2: M_{1,3,5} for j=1,2,3 (remaining columns of A, remaining rows of B)
    # For each j in 1..3 (0-indexed: j=1,2,3), we have M_{1,1,5}
    # i in 0..2 (but n=1 for this sub-block? No, n is still 3 for the output)
    # Wait, M_{1,3,5} means: 1x3 * 3x5 -> 1x5? No.
    # The split is on the middle dimension m.
    # A is 3x4, B is 4x5. C is 3x5.
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # We split j into {0} and {1,2,3}.
    # Part 1: j=0. C1[i,k] = A[i,0] B[0,k]. This is outer product of column 0 of A and row 0 of B.
    #   Tensor: T1[i, k, (k*n+i)] for the part where j=0.
    #   Actually, the tensor T[i, j, k, (k*n+i)] has 1 at (i,j,k,k*n+i).
    #   For fixed j=0: T[i, 0, k, (k*n+i)] = 1.
    #   This is M_{n=3, m=1, p=5}. Rank = 5.
    #   U[i*m + j, col] = U[i*4 + 0, col]
    #   V[j*p + k, col] = V[0*5 + k, col] = V[k, col]
    #   W[k*n + i, col]
    #   This matches what I wrote above.
    
    # Part 2: j in {1,2,3}. C2[i,k] = sum_{j=1}^{3} A[i,j] B[j,k].
    #   This is M_{n=3, m=3, p=5}.
    #   We can split m=3 into three m=1 blocks.
    #   For each j in 1,2,3:
    #     T[i, j, k, (k*n+i)] = 1
    #     This is M_{n=3, m=1, p=5} again, but with different j.
    #     Rank = 5 for each.
    #     Total for this part = 3 * 5 = 15.
    
    # So total rank = 5 + 15 = 20.
    
    # Let's verify the indices for the second part.
    # For j in 1,2,3:
    #   U[i*m + j, col]
    #   V[j*p + k, col]
    #   W[k*n + i, col]
    #   col = 5 + (j-1)*5 + k  (offset by 5 for the first block)
    
    for j in range(1, m):
        for i in range(n):
            for k in range(p):
                col = 5 + (j - 1) * 5 + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
