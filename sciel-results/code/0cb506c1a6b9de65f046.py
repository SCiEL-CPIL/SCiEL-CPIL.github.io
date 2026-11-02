"""
Exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
1. Use the known rank-19 decomposition of the 3x3x3 multiplication tensor (Strassen-like).
2. Extend it to 3x3x7 by using the block structure:
   - The multiplication of (3x3) * (3x7) can be seen as three (3x3) * (3x3) multiplications
     plus one (3x3) * (3x1) multiplication.
   - However, a more direct approach: the tensor T_{3,3,7} has indices (i,j,k) where
     i in 0..2 (rows of result), j in 0..2 (cols of A / rows of B), k in 0..6 (cols of B).
   - We can partition the 7 columns of B into three groups: {0,1,2}, {3,4,5}, {6}.
   - For the first two groups (3 cols each), we use the rank-19 decomposition of T_{3,3,3}.
   - For the last group (1 col), we use the standard rank-9 decomposition of T_{3,3,1}.
   - Total rank: 19 + 19 + 9 = 47.

But wait, there's a better way. The known minimal rank for 3x3x3 is 19 (Strassen).
For 3x3x7, we can also try to find a single decomposition that exploits the full structure.

However, the block construction gives rank 47, which is better than 63.

Let me verify the Strassen decomposition for 3x3x3. The standard Strassen decomposition
for 3x3x3 actually has rank 19. Let me recall the specific tensors.

Actually, the minimal rank of the 3x3x3 matrix multiplication tensor is 19, and there
is a known explicit decomposition. Let me construct it.

The Strassen decomposition for 3x3x3:
The tensor T_{i,j,k} = delta_{i+k, j} (no, that's not right for matrix multiplication).

For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
T_{i,j,k} = 1 if j == k' (the inner dimension), else 0, but actually:
T_{(i,j,k)} = 1 when the multiplication uses A[i,j] * B[j,k].

So the tensor is T_{i,j,k} = 1 for all valid i,j,k, with the constraint that it's a
multiplication tensor. The tensor is T[i,j,k] = 1 if we're multiplying A[i,j] by B[j,k].

Wait, I need to be more careful. The matrix multiplication tensor T has:
T[i, j, k] = 1 if and only if we're using A[i,j] * B[j,k] to compute C[i,k].

So for each (i,j,k), T[i,j,k] = 1. The tensor is the "multiplication" tensor where
each term A[i,j]*B[j,k] contributes to C[i,k].

The rank-19 Strassen decomposition for 3x3x3:
This is well-known but the explicit coefficients are complex. Let me use a different approach.

Actually, for the purpose of this task, let me just use the block construction:
- Split the 7 columns of B into three blocks: B1 (cols 0-2), B2 (cols 3-5), B3 (col 6).
- For B1 and B2, use the rank-19 decomposition of 3x3x3.
- For B3, use the rank-9 schoolbook decomposition of 3x3x1.
- Total rank: 19 + 19 + 9 = 47.

But I need the explicit rank-19 decomposition. Let me look up or derive it.

The Strassen decomposition for 3x3x3 has 19 terms. The standard one is:

Actually, let me just implement the block construction with schoolbook for each block.
That gives 9 + 9 + 9 = 27, which is still better than 63.

Wait, no. For 3x3x3, schoolbook is rank 27. For 3x3x1, schoolbook is rank 9.
So block construction: 27 + 27 + 9 = 63. That's the same as the original.

I need to use Strassen for the 3x3x3 blocks to get 19 + 19 + 9 = 47.

Let me hardcode the Strassen decomposition for 3x3x3. The rank-19 decomposition
of the 3x3x3 multiplication tensor:

The 19 tensors for Strassen's 3x3x3 decomposition (one common form):

Actually, this is getting complex. Let me just return the schoolbook decomposition
with rank 63, which is guaranteed correct, and hope for a better approach.

Given the constraints, I'll return the schoolbook decomposition.
"""

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63
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
