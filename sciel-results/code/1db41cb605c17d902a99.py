"""Matrix multiplication tensor decomposition for 2x3 * 3x10.

The tensor T for (n,m,p) matrix mult has T[i*m+j, j*p+k, k*n+i] = 1.
Schoolbook rank = n*m*p = 60. Lower bound = max(nm,mp,pn) = 30.

We attempt a rank-30 construction using the following idea:
For each (j,k) pair (3*10=30 terms), we create a term where:
- U picks out a linear combination of A[0,j] and A[1,j]
- V picks out B[j,k]
- W distributes to C[0,k] and C[1,k]

Specifically, for each j in {0,1,2} and k in {0,...,9}:
  Term A: U has 1 at (0*m+j), V has 1 at (j*p+k), W has 1 at (k*n+0)
  Term B: U has 1 at (1*m+j), V has 1 at (j*p+k), W has 1 at (k*n+1)

That's 60 terms (schoolbook). To get 30, we need to merge.

Key insight: W can have entries in multiple positions. So for each (j,k), we can use ONE term where:
  U[0*m+j] = 1, U[1*m+j] = 1  (but this gives A[0,j]+A[1,j], not separate)

That doesn't work directly. Let me try:
  For (j,k): U[0*m+j]=1, U[1*m+j]=0, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=0
  For (j,k): U[0*m+j]=0, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=0, W[k*n+1]=1

Still 60. The structure seems to require 60.

However, we can try: for each j and k, use one term with:
  U[0*m+j] = 1, U[1*m+j] = -1, V[j*p+k] = 1, W[k*n+0] = 1, W[k*n+1] = -1
  This gives (A[0,j]-A[1,j])*B[j,k] contributing to C[0,k]-C[1,k]

And another: U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=1
  This gives (A[0,j]+A[1,j])*B[j,k] contributing to C[0,k]+C[1,k]

Still 60 terms. I cannot find a rank-30 construction.

Falling back to schoolbook rank 60.
"""

import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
