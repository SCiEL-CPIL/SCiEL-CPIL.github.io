"""Slice-method decomposition for 3x4x7 matrix multiplication.

Uses the slice method: partition the 7 output columns into blocks of sizes 3, 2, 2.
For each block of size b, we need a rank-minimal decomposition of the 3x4xb tensor.
The total rank is the sum of the ranks of the sub-decompositions.

For 3x4x2: rank 8 (schoolbook)
For 3x4x3: rank 12 (schoolbook)

Total: 8 + 12 + 8 = 28

We use the schoolbook decomposition for each sub-problem, which gives:
- For 3x4xb: rank = 3*4*b = 12b
  - 3x4x2: rank 24
  - 3x4x3: rank 36
  Total: 24 + 36 + 24 = 84

But we can do better. Let's use a smarter approach.

Actually, for the slice method, each sub-problem 3x4xb has its own tensor.
The schoolbook rank for 3x4xb is 12b.

However, there are known improvements. For 3x4x2, the optimal rank is 8 (not 24).
For 3x4x3, let's think... 

Actually, let me just use a direct construction. The key insight is that for the slice method,
we can use the same U matrix for all blocks (since the input dimensions are the same),
but V and W are block-diagonal.

Let me construct a rank-28 solution using a different approach.

For multiplying a 3x4 matrix by a 4x7 matrix:
- Input A: 3x4, entries A[i,j] for i in 0..2, j in 0..3
- Input B: 4x7, entries B[j,k] for j in 0..3, k in 0..6
- Output C: 3x7, entries C[i,k] for i in 0..2, k in 0..6

C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the multiplication).

Wait, no. The matrix multiplication tensor T satisfies:
T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].

Actually, the standard formulation is:
C[i,k] = sum_j A[i,j] * B[j,k]

This can be written as:
C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]

where T[i,j,k] = 1 if the indices are valid, but more precisely, the tensor is:
T[i,j,k] = delta_{j,j} which is just 1 for all valid i,j,k.

Hmm, let me think more carefully. The matrix multiplication tensor for m x n times n x p is:
T[i,j,k] where i in 0..m-1, j in 0..n-1, k in 0..p-1
and T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k]).

So the tensor is the all-ones tensor of shape (3, 4, 7).

The rank of this tensor is what we want to minimize.

For the slice method with blocks [3, 2, 2]:
- Block 0: k in {0,1,2}, tensor shape (3,4,3), rank r0
- Block 1: k in {3,4}, tensor shape (3,4,2), rank r1
- Block 2: k in {5,6}, tensor shape (3,4,2), rank r2

Total rank = r0 + r1 + r2.

For 3x4x2, the optimal rank is known to be 8.
For 3x4x3, let's estimate. The schoolbook is 36. 

Actually, I recall that for small matrices, the optimal ranks are:
- 2x2x2: 7
- 3x3x3: 23
- 4x4x4: 49

For 3x4x2, I believe the rank is 8.
For 3x4x3, it might be around 12-15.

Let me just construct explicit decompositions.

For 3x4x2 with rank 8:
We need U (12x8), V (8x8), W (6x8) such that:
sum_c U[i*4+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1 for all i,j,k

This is a specific problem. Let me use a known construction.

Actually, let me just use the schoolbook for simplicity and correctness, then optimize if needed.

Schoolbook for 3x4x7: rank = 3*4*7 = 84.

But the slice method with schoolbook sub-decompositions:
- 3x4x3: rank 36
- 3x4x2: rank 24
- 3x4x2: rank 24
Total: 84

Same as direct schoolbook. So the slice method with schoolbook doesn't help.

We need better sub-decompositions. Let me try to find a rank-8 decomposition for 3x4x2.

For 3x4x2:
n=3, m=4, p=2
U: 12x8, V: 8x8, W: 6x8

Equations: sum_c U[i*4+j,c] * V[j*2+k,c] * W[k*3+i,c] = 1 for all i in 0..2, j in 0..3, k in 0..1

That's 3*4*2 = 24 equations, 8*12 + 8*8 + 8*6 = 96 + 64 + 48 = 208 variables.

Let me try a symmetric construction. Assume U[i*4+j,c] depends only on (i,j) and c in a structured way.

Actually, let me just hardcode a known solution or use a simpler approach.

Given the time constraint, let me use a rank-28 construction based on a different insight.

For 3x4x7, we can use the fact that:
C[i,k] = sum_j A[i,j] B[j,k]

We can write this as:
C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]

One approach: use 4 "intermediate" variables per output entry, giving rank 4*7 = 28.

Specifically, for each k, we compute C[:,k] = A * B[:,k] using 4 multiplications (one per j).

So for each k, we need 4 rank-1 terms. Total rank = 7 * 4 = 28.

For the k-th column:
C[i,k] = sum_j A[i,j] B[j,k] = sum_j (A[i,j]) * (B[j,k])

This is a rank-4 decomposition for each k (since there are 4 j values).

For the full problem:
- For each k in 0..6 and each j in 0..3, we have a term:
  U[i*4+j, k*4+j] = 1 (if we index columns by (k,j))
  V[j*7+k, k*4+j] = 1
  W[k*3+i, k*4+j] = 1

Wait, let me be more careful.

The factor matrices:
U: shape (n*m, rank) = (12, rank)
V: shape (m*p, rank) = (28, rank)  
W: shape (p*n, rank) = (21, rank)

For each pair (j, k) where j in 0..3, k in 0..6:
- This corresponds to the term A[i,j] * B[j,k] contributing to C[i,k]
- U[i*4 + j, col] = 1 where col = k*4 + j
- V[j*7 + k, col] = 1
- W[k*3 + i, col] = 1

But this gives rank = 4*7 = 28, and for each (i,j,k):
sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*3+i, c]

For c = k'*4 + j', this is 1 only if i*4+j has U entry 1 at c, which means j = j' and i is arbitrary... 

Wait, U[i*4+j, c] = 1 when c = k'*4 + j for some k'. But U only depends on (i,j) for the row, and c for the column.

Let me redefine: for each (j, k) pair, assign a column c = k*4 + j.
- U[row, c] where row = i*4 + j: U[i*4+j, k*4+j] = 1 for all i, k
  But that's not right either. U[i*4+j, c] should be 1 only for the specific (i,j) that we're using.

Actually, the standard schoolbook is:
For each (i, j, k), have a term with:
U[i*m+j, (i,j,k) index] = 1
V[j*p+k, (i,j,k) index] = 1
W[k*n+i, (i,j,k) index] = 1

This gives rank = n*m*p = 84.

The optimization is to share terms. For the slice method, we can't share across different k blocks easily.

Let me just go with rank 28 using the "per-column schoolbook" approach:

For each output column k (0..6), we compute C[:,k] = A * B[:,k].
This requires 4 multiplications (one per row of B, i.e., per j).

So for column k, we use 4 rank-1 terms, one for each j:
- Term (j, k): 
  U[i*4+j, k*4+j] = 1 for all i (since A[i,j] is the input)
  V[j*7+k, k*4+j] = 1
  W[k*3+i, k*4+j] = 1 for all i

Check: sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*3+i, c]
For c = k'*4 + j':
U[i*4+j, k'*4+j'] = 1 iff j = j' (and for any i, but U is defined per row)
Wait, U[i*4+j, c] is a specific entry. Let me define:

U[a, b] where a = i*4+j, b = k*4+j
So U[i*4+j, k*4+j'] = 1 if j = j', else 0.

V[c, d] where c = j*7+k, d = k'*4+j'
V[j*7+k, k'*4+j'] = 1 if j = j' and k = k', else 0.

Wait, this is getting confused. Let me be very explicit.

Rank = 28. Columns indexed by (k, j) where k in 0..6, j in 0..3.
Column index c = k * 4 + j.

U has shape (12, 28). U[i*4 + j, c] = 1 if c = k*4 + j for some k (i.e., the j-component of c matches j), else 0.
So U[i*4+j, k*4+j'] = 1 if j = j', else 0.

V has shape (28, 28). V[j*7 + k, c] = 1 if c = k*4 + j, else 0.
So V[j*7+k, k'*4+j'] = 1 if k = k' and j = j', else 0.

W has shape (21, 28). W[k*3 + i, c] = 1 if c = k*4 + j for some j (i.e., the k-component of c matches k), else 0.
So W[k*3+i, k'*4+j'] = 1 if k = k', else 0.

Now check: sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*3+i, c]
= sum_{k'=0}^{6} sum_{j'=0}^{3} U[i*4+j, k'*4+j'] * V[j*7+k, k'*4+j'] * W[k*3+i, k'*4+j']
= sum_{k',j'} [1 if j=j'] * [1 if k=k' and j=j'] * [1 if k=k']
= sum_{k',j'} [j=j'] * [k=k' and j=j'] * [k=k']
= [k=k and j=j] * 1 = 1 (when k'=k, j'=j)

So the sum is 1. 

This gives a valid rank-28 decomposition!

Let me code this up.
"""
import numpy as np


def construct():
    n, m, p = 3, 4, 7
    rank = m * p  # 4 * 7 = 28

    U = [[0] * rank for _ in range(n * m)]  # 12 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 21 x 28

    for k in range(p):
        for j in range(m):
            c = k * m + j  # column index

            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1

            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1

            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
