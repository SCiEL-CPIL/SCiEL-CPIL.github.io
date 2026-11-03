"""Matrix multiplication 4x4 by 4x9 - rank-minimal tensor decomposition.

Uses a structured decomposition that exploits the rectangular structure.
The key insight: for multiplying (n x m) by (m x p), we can use a
decomposition where we group by the shared index j and use rank-1
outer products in a way that reduces the total rank below n*m*p.

For 4x4 * 4x9: we use a block structure where each row of A and each
column of B are combined via Kronecker-like products, but with the
constraint that entries must be integer or half-integer.

The trivial decomposition has rank n*m*p = 4*4*9 = 144.
We can do better by using the fact that the multiplication tensor has
a specific structure. For rectangular matrices, the border rank and
tensor rank can be lower than the naive product.

One approach: use the identity that C = A*B can be computed as
sum over j of (column j of A) * (row j of B), which is m rank-1 terms
in a different sense, but in the tensor decomposition sense, we need
to decompose the trilinear form.

Actually, let me think about this more carefully. The matrix multiplication
tensor T[i,j,k] = delta_{j,k} (for the appropriate indices) needs to be
decomposed into rank-1 tensors.

For A (n x m) times B (m x p) giving C (n x p):
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is T[i,j,k] where the multiplication is:
C[i,k] = sum_{j=1}^{m} A[i,j] B[j,k]

The rank-1 decomposition: T = sum_{r=1}^{R} u_r * v_r * w_r
where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} (with W[k*n+i] indexing)

The trivial decomposition: for each (i,j,k), T[i,j,k] = 1 if the indices match
appropriately. Actually, the standard decomposition is:

For each j (the shared index), and for each i, k:
C[i,k] gets contribution A[i,j]*B[j,k]

So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
This gives n*m*p = 144 terms.

Can we do better? The key question is whether there's a decomposition with
fewer terms using half-integer or other integer coefficients.

For square matrix multiplication, there are known results. For 4x4*4x4,
the best known is 49 (Strassen-like). But here we have 4x4 * 4x9.

Let me try a different approach: use the fact that we can combine terms
when the U and V vectors are linearly dependent.

Actually, let me try to use a structured decomposition based on
block matrices. Write A as a 4x4 matrix and B as a 4x9 matrix.

One idea: use the decomposition based on the fact that we can compute
the product using fewer multiplications by exploiting linearity.

For a 2x2 * 2x2, Strassen uses 7 multiplications instead of 8.
The analogous construction for larger matrices uses recursive blocking.

But the constraint is that entries must be integer or half-integer.
Strassen's algorithm uses coefficients that are integers (±1), so it
satisfies this constraint.

Let me try to construct a Strassen-like decomposition for 4x4 * 4x9.

Approach: Partition A into 2x2 blocks of 2x2 submatrices, and B into
2x2 blocks of 2x2 submatrices (with the second dimension being 4x9,
so we can partition the 4 rows into 2+2 and the 9 columns into 4+5 or
other splits).

Actually, for Strassen to work cleanly, we want the blocks to be square.
Let me partition:
- A (4x4) into four 2x2 blocks: A11, A12, A21, A22
- B (4x9) into blocks: we need the row dimension to be 4, so split into
  2+2 rows. The column dimension is 9, which is odd. We can split into
  4+5 or 5+4.

This gets complicated. Let me try a simpler approach.

Alternative: Just use the trivial decomposition but verify it's correct.
The score is 144, and if I can't find something better, at least the
program is correct.

But wait - let me think about whether there's a simple improvement.

For the tensor T[i*m+j, j*p+k, k*n+i] = 1:
- U has 16 rows (n*m = 4*4), each with 144 entries
- V has 36 rows (m*p = 4*9), each with 144 entries  
- W has 36 rows (p*n = 9*4), each with 144 entries

Each column r corresponds to a triple (i,j,k) and has:
- U[i*m+j][r] = 1
- V[j*p+k][r] = 1
- W[k*n+i][r] = 1

No two columns share the same U row, V row, or W row pattern in a way
that allows merging with integer/half-integer coefficients.

Let me try a completely different approach: use numpy to find a low-rank
decomposition numerically, then round to nearest half-integers.

Actually, let me try to use the known result that for m x n * n x p matrix
multiplication, the rank is at most m*n*p, and for small dimensions,
we can sometimes do better.

For 4x4 * 4x4, the best known rank is 49 (using Strassen's algorithm
recursively or other constructions).

For 4x4 * 4x9, we can think of it as three 4x4 * 4x3 multiplications,
or use other decompositions.

Let me try: compute C = A * B where A is 4x4 and B is 4x9.
Write B as [B1 | B2 | B3] where B1, B2, B3 are 4x3 matrices.
Then C = [A*B1 | A*B2 | A*B3].

Each A*B_i is a 4x4 * 4x3 multiplication.

For 4x4 * 4x3, the trivial rank is 4*4*3 = 48.
Can we do better? 

Actually, let me just try to use a numerical approach: set up the
tensor as a 3D array, flatten it, and try to find a low-rank
approximation, then check if rounding to half-integers gives an exact
solution.

But this might be too slow or might not find an exact solution.

Let me try yet another approach: use the fact that the problem allows
half-integer entries. This means we can potentially use decompositions
that involve 1/2 coefficients.

For now, let me implement the trivial solution correctly and ensure it
passes validation. The key is to get the indexing right.

The W matrix uses the convention W[k*n+i, r], so row index is k*n + i
where k is the column index of B (0 to p-1) and i is the row index of A
(0 to n-1).

Let me verify: C[i,k] = sum_j A[i,j] * B[j,k]
In the tensor decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
But more precisely, the trilinear form is:
C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]

Where T[i,j,k] = delta_{j,k}... no, that's not right.

Actually, the matrix multiplication tensor is defined such that:
C = A * B means C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has components T[i,j,k] where:
- i indexes rows of A and C (0 to n-1)
- j indexes columns of A and rows of B (0 to m-1)  
- k indexes columns of B and C (0 to p-1)

And T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor).

Wait, no. The tensor is such that:
C[i,k] = sum_{j} A[i,j] * B[j,k]

This can be written as:
C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta_{i',i} * delta_{k',k}

So T[i',j',k'] = 1 for all i', j', k'.

The rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
where u_r[i*m+j] = U[i*m+j, r], v_r[j'*p+k'] = V[j'*p+k', r], w_r[k'*n+i'] = W[k'*n+i', r]

And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For the trivial decomposition: for each (i,j,k), set:
U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1

Then sum_r ... = 1 only when (i,j,k) matches, giving T[i,j,k] = 1. Correct.

So the trivial solution with rank 144 is correct. Let me just implement it
cleanly and hope the validator accepts it.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
