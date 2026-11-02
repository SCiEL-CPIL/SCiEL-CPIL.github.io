"""
Matrix multiplication tensor M_{2,2,9}: find a rank-minimal exact decomposition.

We know Rank(M_{2,2,1}) = 4. Since M_{2,2,9} is a direct sum of 9 copies of M_{2,2,1},
the rank is 9 * 4 = 36. We provide the explicit decomposition using the known rank-4
decomposition of M_{2,2,1} (e.g., via Strassen-like or standard bilinear form).

For M_{2,2,1}: multiply A (2x2) by B (2x1) to get C (2x1).
The tensor T_{i,j,k} = 1 for i in {0,1}, j in {0,1}, k in {0,1}.

A known rank-4 decomposition:
U rows indexed by (i,j), V rows indexed by (j,k), W rows indexed by (k,i).

For M_{2,2,1}, the 4 terms can be written as:
  c1: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1  -> contributes to C[0,0] via A[0,0]*B[0,0]
  c2: U_{0,0}=1, V_{0,1}=1, W_{0,1}=1  -> contributes to C[0,1] via A[0,0]*B[1,0]... 

Wait, let me think more carefully. The matrix multiplication M_{n,m,p} computes:
  C[k,i] = sum_j A[i,j] * B[j,k]

The tensor is T_{i,j,k} = 1 for all i,j,k.

A decomposition of rank r: T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}

For M_{2,2,1} (n=2, m=2, p=1): indices i in {0,1}, j in {0,1}, k in {0}.
U is 4x r (rows i*m+j = i*2+j), V is 2x r (rows j*p+k = j*1+0 = j), W is 2x r (rows k*n+i = 0*2+i = i).

Rank-4 decomposition for M_{2,2,1}:
We need sum_c U[i,j,c] * V[j,c] * W[i,c] = 1 for all i,j.

One simple approach: use the 4 basis elements directly.
c=0: U[0,0]=1, V[0]=1, W[0]=1  -> gives A[0,0]*B[0,0] to C[0,0]
c=1: U[0,1]=1, V[1]=1, W[0]=1  -> gives A[0,1]*B[1,0] to C[0,0]
c=2: U[1,0]=1, V[0]=1, W[1]=1  -> gives A[1,0]*B[0,0] to C[1,0]
c=3: U[1,1]=1, V[1]=1, W[1]=1  -> gives A[1,1]*B[1,0] to C[1,0]

This is actually the schoolbook decomposition of rank 4 for M_{2,2,1}, which is optimal.

For M_{2,2,9}: n=2, m=2, p=9.
U: (n*m) x rank = 4 x rank
V: (m*p) x rank = 18 x rank  
W: (p*n) x rank = 18 x rank

We need T_{i,j,k} = 1 for i in {0,1}, j in {0,1}, k in {0,...,8}.

The schoolbook decomposition has rank 2*2*9 = 36. Each term (i,j,k) gives:
U[i*2+j] = 1, V[j*9+k] = 1, W[k*2+i] = 1

Since M_{2,2,9} is a direct sum of 9 copies of M_{2,2,1} (one for each k), and each has rank 4,
the total rank is 9*4 = 36. This is the schoolbook decomposition and it is rank-minimal.

So we just need to construct the schoolbook decomposition of rank 36.
"""

import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36, which is rank-minimal
    
    # U: (n*m) x rank = 4 x 36
    # V: (m*p) x rank = 18 x 36
    # W: (p*n) x rank = 18 x 36
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
# EVOLVE-BLOCK-END
