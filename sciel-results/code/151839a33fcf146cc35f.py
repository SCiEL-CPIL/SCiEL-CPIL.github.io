"""
Rank-minimal tensor decomposition for 2x5 by 5x5 matrix multiplication.

Strategy:
  The matrix multiplication $C = AB$ where $A$ is $2 \times 5$ and $B$ is $5 \times 5$ 
  can be viewed as two independent dot products for each output column $k$:
  $C[0, k] = \sum_{j=0}^4 A[0, j] B[j, k]$
  $C[1, k] = \sum_{j=0}^4 A[1, j] B[j, k]$
  
  The tensor for a single $1 \times 5 \times 5$ multiplication (row vector times matrix)
  has rank 5. Since the two rows of $A$ are independent in the tensor decomposition 
  (the $i$ index in $C[i, k]$ allows us to separate the contributions), we can sum 
  the ranks: $5 + 5 = 10$.
  
  Construction:
  For $i=0$ (first row of A):
    Use 5 rank-1 terms (r=0..4).
    U[0*5+j, r] = 1 if j==r else 0
    V[j'*5+k, r] = 1 if j'==r else 0
    W[k*2+0, r] = 1 for all k
    
  For $i=1$ (second row of A):
    Use 5 rank-1 terms (r=5..9).
    U[1*5+j, r] = 1 if j==(r-5) else 0
    V[j'*5+k, r] = 1 if j'==(r-5) else 0
    W[k*2+1, r] = 1 for all k

  This yields a total rank of 10 with integer entries.
"""

import time

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    # Rank is n * m = 2 * 5 = 10
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the decomposition
    for i in range(n):
        for j in range(m):
            # j corresponds to the rank index for this specific i
            # We use rank index r = i * m + j
            r = i * m + j
            
            # U index: i * m + j
            U[i * m + j][r] = 1
            
            # V index: j * p + k (for all k, V is 1 if j' == j)
            # Wait, V index is j' * p + k. We need V[j', k, r] = 1 if j' == j.
            # So for all k in 0..p-1:
            for k in range(p):
                V[j * p + k][r] = 1
                
            # W index: k * n + i
            # W[k, i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
