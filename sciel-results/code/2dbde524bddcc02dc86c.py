"""
Rank-48 decomposition for 3x4 * 4x4 matrix multiplication.

Uses a partitioned Strassen-like approach:
- Split A (3x4) into top 2 rows (2x4) and bottom row (1x4).
- Split B (4x4) into left 2 cols (4x2) and right 2 cols (4x2).
- Compute 4x4 products via Strassen's 7 multiplications.
- Compute 1x4 * 4x4 via 4 multiplications (rank 4).
- Total: 7*4 + 4 = 32 multiplications? No, Strassen gives rank 7 per 4x4 product.
  Actually for 4x4 = (2x2 blocks), Strassen gives 7 multiplications of 2x2, each rank 4, so 7*4=28 per 4x4 product.
  We need two 4x4 products: 2*28 = 56? That's worse.

Let me reconsider. Use the fact that we only need a 3x4 output.

Alternative: Use the known rank 48 decomposition from literature for 3x4*4x4.
Since I don't have it memorized, I'll implement a general rank-reduction approach:

Partition A into [A1; A2] where A1 is 2x4 and A2 is 1x4.
Partition B into [B1 | B2] where B1 is 4x2 and B2 is 4x2.
Output C = [A1*B1, A1*B2; A2*B1, A2*B2].

We can compute:
- A1*B1: 2x4 * 4x2 = 2x2, rank 4
- A1*B2: 2x4 * 4x2 = 2x2, rank 4
- A2*B1: 1x4 * 4x2 = 1x2, rank 2
- A2*B2: 1x4 * 4x2 = 1x2, rank 2
Total: 4+4+2+2 = 12? No, these are the sizes, not ranks.

Actually, the schoolbook for each sub-product:
- 2x4 * 4x2: 2*4*2 = 16
- 1x4 * 4x2: 1*4*2 = 8
Total: 16+16+8+8 = 48. That's the partitioned schoolbook with rank 48!

So splitting gives rank 48 vs schoolbook 48 (3*4*4=48). Same rank.

But wait, 3*4*4 = 48. The partitioned version also gives 48. No improvement.

I'll return the schoolbook decomposition which is correct and has rank 48.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0] * rank for _ in range(m * p)]  # 16 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48

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
