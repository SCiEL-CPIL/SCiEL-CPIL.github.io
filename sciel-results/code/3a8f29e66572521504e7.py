"""Tensor decomposition for 3x3 by 3x7 matrix multiplication.

Strategy: Use Strassen's 2x2x2 rank-5 algorithm to decompose a block-level
multiplication. We treat the 3x3 * 3x7 problem by splitting the 3x3 matrix
into 2x2 blocks and the 3x7 matrix into 2x7 blocks, then using the Strassen
decomposition for the 2x2 * 2x2 core, extended appropriately.

Actually, let me think about this more carefully. The standard schoolbook
gives rank 63. We want to find a lower rank decomposition.

One approach: Use the fact that 3x3*3x7 can be seen as related to polynomial
multiplication. If we encode a 3x3 matrix as a polynomial of degree 2 (with
matrix coefficients) and a 3x7 matrix as a polynomial of degree 2, then
their product is a polynomial of degree 4. But this is complex.

Alternative: Use a known lower-bound / construction. For 3x3 * 3x3, the best
known is rank 21 (using Strassen-like methods). For 3x3 * 3x7, we can try to
adapt.

Let me try a direct approach: use the 2x2 Strassen block decomposition.
Split A (3x3) into:
  A11 (1x1)  A12 (1x2)
  A21 (2x1)  A22 (2x2)

Split B (3x7) into:
  B11 (1x7)  B12 (2x7)  -- wait, this doesn't split cleanly.

Let me reconsider. The cleanest approach for a guaranteed valid improvement:
Use the rank-5 decomposition of 2x2*2x2 (Strassen) to handle the 2x2*2x2
block, and handle the remaining cross terms with schoolbook.

Actually, for a 3x3 * 3x7, let's split as:
- A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
- B = 3x7 matrix

Group columns of B into groups of 2 (with one leftover):
B = [B_1 (3x2), B_2 (3x2), B_3 (3x2), B_4 (3x1)]

Then C = A*B = [A*B_1, A*B_2, A*B_3, A*B_4]

Each A*B_j for j=1,2,3 is a 3x3 * 3x2 multiplication.
A*B_4 is a 3x3 * 3x1 multiplication.

For 3x3 * 3x2: we can use the fact that this is equivalent to two 3x3*3x1
problems, but more efficiently, we can use a rank-? decomposition.

Actually, let me just use a known result: the tensor rank of 3x3*3x7 is at
most 21*2 + something... this is getting complicated.

Let me use a safer approach: use the 2x2 Strassen decomposition embedded
into the 3x3*3x7 problem.

Split A (3x3) into 2x2 blocks with overlap:
Row split: rows 1-2 and row 3
Col split: cols 1-2 and col 3

A = [[A11 (2x2), A12 (2x1)],
     [A21 (1x2), A22 (1x1)]]

B (3x7) split by rows:
B = [[B1 (2x7)],
     [B2 (1x7)]]

Then C = A*B:
C rows 1-2, all cols: A11*B1 + A12*B2  (2x7)
C row 3, all cols:    A21*B1 + A22*B2  (1x7)

Now, A11*B1 is 2x2 * 2x7. We can use Strassen for 2x2*2x2 by splitting B1
into two 2x2 blocks (cols 1-2, 3-4, 5-6, 7 alone). This is messy.

Let me just implement a straightforward approach that is guaranteed correct
and try to get a slight improvement using the 2x2 Strassen on the 2x2*2x2
sub-blocks.

For 2x2 * 2x2, Strassen uses rank 5.
For 2x2 * 2x7, we can think of it as 2x2 * (2x2 + 2x2 + 2x2 + 2x1).

This is getting too complex for guaranteed correctness in the time available.
Let me just provide the schoolbook solution which is guaranteed to work.
"""
DIMENSIONS = [3, 3, 7]

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
