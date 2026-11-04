"""Strassen-based rank-84 exact tensor decomposition for 3x5x6 matrix multiplication."""

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    # We decompose the 3x5x6 multiplication by splitting the first dimension into 1+2
    # and applying Strassen on the 2x5x6 sub-block.
    #
    # Matrix A is 3x5, B is 5x6, C is 3x6.
    # Split A into A_top (1x5) and A_bot (2x5).
    # C_top = A_top * B  (1x6)
    # C_bot = A_bot * B  (2x6)
    #
    # C_top uses rank 5 (standard)
    # C_bot uses rank 7 (Strassen 2x2x2 on 2x5x5 sub-blocks, then multiply by 6 columns)
    #   Wait, Strassen on 2x2x2 gives rank 7 for 2x2x2.
    #   For 2x5x6, we can split the middle dimension into blocks.
    #
    # Let me reconsider. The known rank for 3x5x6 is 84 = 5 + 7*... 
    # Actually 84 = 5 + 7*11? No.
    #
    # Let's think differently. The 3x5x6 tensor can be seen as:
    #   rank(3,5,6) <= rank(1,5,6) + rank(2,5,6)
    #   rank(1,5,6) = 5 (standard)
    #   rank(2,5,6) <= ?
    #
    # For 2x5x6: split the 5 dimension into 2+2+1:
    #   rank(2,5,6) <= rank(2,2,6) + rank(2,2,6) + rank(2,1,6)
    #   rank(2,2,6) = 6 * rank(2,2,2) = 6*7 = 42? That's too high.
    #
    # Alternative: split the 6 dimension into 3+3:
    #   rank(2,5,6) <= rank(2,5,3) + rank(2,5,3)
    #   rank(2,5,3) <= ?
    #
    # Let me use a different approach. The known result is that M(3,5,6) <= 84.
    # One construction:
    #   M(3,5,6) <= M(1,5,6) + M(2,5,6)
    #   M(1,5,6) = 5
    #   M(2,5,6) <= M(2,2,6) + M(2,2,6) + M(2,1,6)
    #   M(2,1,6) = 6
    #   M(2,2,6) = 6 * M(2,2,1) = 6*2 = 12? No, M(2,2,1)=2.
    #   So M(2,2,6) = 12, M(2,1,6) = 6, total = 5 + 12 + 12 + 6 = 35? That's not 84.
    #
    # I think I'm overcomplicating this. Let me just implement the standard
    # Strassen-based decomposition that gives rank 84.
    #
    # Actually, let me try a direct approach: use rank-7 Strassen for 2x2x2
    # and standard multiplication elsewhere.
    #
    # 3x5x6: split 3 = 1 + 2, 5 = 2+2+1, 6 = 3+3
    # This gets complicated. Let me just use the known rank-84 construction.
    #
    # For now, let me implement a correct but possibly not optimal decomposition
    # and verify it works.

    # Standard decomposition: rank = n*m*p = 90
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k), V[j*p+k, c] = 1, W[k*n+i, c] = 1
    rank = n * m * p
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
