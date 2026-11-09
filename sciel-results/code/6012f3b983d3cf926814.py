"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7

    # We'll build a decomposition with rank r = 45.
    # Strategy:
    #   1. Use a 3x3x3 Strassen-like decomposition for the 2x3x3 block (rows 0,1 of A; cols 0,1,2 of B; rows 0,1,2 of C).
    #      This uses 10 multiplications.
    #   2. The remaining parts are 2x1x3 (rows 0,1 of A; col 3 of B; rows 0,1,2 of C) and 2x4x1 (rows 0,1 of A; cols 0-3 of B; row 3 of C).
    #      We handle these with 10 + 8 + ... multiplications.
    #
    # Actually, let's use a simpler approach:
    #   Total rank = n*m*p = 56 (schoolbook)
    #   We want to reduce this.
    #
    # Let's use the fact that we can combine some multiplications.
    #
    # Alternative approach: use the rank-40 decomposition if we can find one.
    #
    # For now, let's use a rank-48 decomposition:
    #   - 2x4x7 = (2x2x7) + (2x2x7) + (2x2x1) + (2x2x1) + (2x2x1)
    #   But this doesn't help directly.
    #
    # Let's just use the schoolbook decomposition but with a slightly better structure.
    # Actually, the task asks for rank-minimal. The current best known for 2x4x7 is 32.
    #
    # Let me implement a known rank-32 decomposition for 2x4x7.
    #
    # The decomposition uses the following idea:
    #   We can write 2x4x7 = (1x4x7) + (1x4x7) with some sharing.
    #
    # Actually, let's use a concrete rank-35 decomposition that I can verify.
    #
    # I'll use the following approach:
    #   - Split the 4 columns of B into two groups of 2.
    #   - Use Strassen for 2x2x2 blocks (7 multiplications each).
    #   - Handle the remaining parts efficiently.
    #
    # Let me just provide a valid rank-45 decomposition for now.
    #
    # Actually, let's provide the schoolbook solution but with a small optimization.
    # The minimal rank for 2x4x7 is known to be at most 32.
    #
    # I'll implement a rank-32 decomposition based on known results.
    #
    # For a 2x4x7 matrix multiplication, one can use the following rank-32 decomposition:
    #
    # This is complex to derive from scratch. Let me provide a valid higher-rank decomposition
    # that is better than 56.
    #
    # Let's use rank = 45.
    #
    # I'll construct it by combining:
    #   - A 2x3x3 block with rank 10 (using a known decomposition)
    #   - The remaining parts with rank 35
    #
    # This is getting complicated. Let me just use the schoolbook solution.
    # The task says "rank-minimal exact tensor decomposition". If I can't find a better one,
    # I'll use the schoolbook.
    
    rank = n * m * p  # 56
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    assert rank == n * m * p
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    print("Construct successful")
