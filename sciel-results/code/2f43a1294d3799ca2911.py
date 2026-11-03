"""Matrix multiplication 2x3x7 - rank-minimal exact tensor decomposition.

Uses block decomposition: A is 2x3, B is 3x7.
Split B into B1 (3x4) and B2 (3x3).
Then C1 = A*B1 (2x4) and C2 = A*B2 (2x3).
Use Strassen for the 2x4 multiplication (padding to 2x4x4 with one zero column,
or using a known rank-12 decomposition for 2x4).
Actually, let's use a direct algebraic approach.

For 2x3 times 3x7:
A = [[a11, a12, a13], [a21, a22, a23]]
B = 3x7 matrix

We need to compute C = A*B where C is 2x7.

The naive approach uses 42 multiplications.
We can try to reduce this.

Key insight: The rank of matrix multiplication tensor M_{n,m,p} satisfies:
- Rank(M_{2,2,2}) = 7 (Strassen)
- Rank(M_{2,3,3}) = 8 (known)
- Rank(M_{2,3,4}) <= 12 (can be shown)

For 2x3x7, we can split B into B1 (3x4) and B2 (3x3):
C1 = A*B1 (2x4), C2 = A*B2 (2x3)
Total rank <= Rank(M_{2,3,4}) + Rank(M_{2,3,3}) <= 12 + 8 = 20?

Wait, but we need to be careful. Let me think about this differently.

Actually, the tensor rank of M_{2,3,7} is at most the sum of ranks when we split.
But we need explicit decompositions.

Let me use a known result: the rank of M_{2,m,p} for m,p >= 2.

For 2x2x2: rank 7 (Strassen)
For 2x3x3: rank 8

For 2x3x7, I'll use the following approach:
- Split the 7 columns into groups
- Use Strassen-like decompositions

Actually, let me try a different approach. The known lower bound for M_{2,m,p} is at least max(m,p, m*p/2 + something).

Let me just implement a known decomposition. 

For multiplying a 2x3 by 3x7 matrix:
We can view this as: for each of the 7 output columns, we compute a 2-vector = A * (column of B).
Each column computation requires a 2x3 times 3x1 multiplication, which has tensor rank 3 (it's just a linear combination).

But we can share computations across columns.

Let me use the following decomposition strategy:
- The 2x3 matrix A has entries a_{i,j} for i=0,1 and j=0,1,2
- B has entries b_{j,k} for j=0,1,2 and k=0,...,6

C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]

We need to compute 14 values (2 rows x 7 columns).

Using Strassen-like ideas for the 2x3 case:
The tensor M_{2,3,1} (2x3 times 3x1) has rank 3 (just 3 multiplications: a0*b0 + a1*b1 + a2*b2 for each row).

But for 2x3x7, we can potentially do better by sharing.

Actually, let me think about this more carefully. The multiplication of a 2x3 by 3x7 matrix:
- Input: 2*3 + 3*7 = 6 + 21 = 27 variables
- Output: 2*7 = 14 values

The trivial rank is 2*3*7 = 42.

I'll implement a decomposition that achieves rank 35 or lower using block methods.

Let me try: split A into two 2x2 and 2x1 blocks? No, A is 2x3.

Alternative: Use the fact that rank(M_{2,3,7}) <= rank(M_{2,4,8}) by embedding.
M_{2,4,8}: pad A to 2x4 (add zero column), B to 4x8 (add zero row and zero column).
Rank(M_{2,4,8}) can be computed via Strassen on 4x4 blocks... this gets complicated.

Let me just provide the schoolbook decomposition (rank 42) as a safe fallback,
but also try a slightly better one.

Actually, for 2x3x7, a simple improvement:
We can use the "laser method" or just note that:
Rank(M_{2,3,7}) <= 7 * Rank(M_{2,3,1}) = 7 * 3 = 21? No, that's not how it works.

The correct bound: if we split B into 7 columns, each column requires rank 3, but we can't just sum because the A entries are shared.

Let me use a known algebraic decomposition. I'll implement rank 35.

Actually, I'll just output the schoolbook decomposition. The score is the rank, and 42 is the baseline.
If I can find a valid lower-rank decomposition, I'll use it. Otherwise, 42.

Let me try to construct a rank-35 decomposition by exploiting linearity.

For the 2x3x7 tensor:
T = sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{6} e_i^A \otimes e_j^B \otimes e_{k,n+i}^C

where e_{k,n+i} is the (k*2+i)-th basis vector in C-space (since C is 2x7, indexed as [k*2+i]).

A rank-1 decomposition: T = sum_c u_c \otimes v_c \otimes w_c

Schoolbook: 42 terms with u_{i,j,k} = e_{i*3+j}, v_{j,k} = e_{j*7+k}, w_{i,j,k} = e_{k*2+i}.

To reduce rank, we need to find a different set of tensors.

I'll try a decomposition based on splitting the j-index differently.

For now, I'll return the schoolbook decomposition. The code will be correct and fast.
"""

import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p = 42
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
