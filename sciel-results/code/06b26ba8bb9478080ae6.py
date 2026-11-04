"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Partition m=5 into [2, 3] and p=5 into [2, 3]
    # Block sizes: (2,2), (2,3), (3,2), (3,3)
    # For each (m_block, p_block) with n=4:
    #   (2,2): schoolbook rank = 4*2*2 = 16 (or use Strassen? No, Strassen is for square matrices)
    #   Actually, let's think about this more carefully.
    #   
    # For multiplying an n x m matrix by an m x p matrix, the tensor rank for schoolbook is n*m*p.
    # With block decomposition, if we split m into m1+m2 and p into p1+p2, we get 4 subproblems:
    #   (n, m1, p1), (n, m1, p2), (n, m2, p1), (n, m2, p2)
    # Each solved independently, total rank = sum of ranks.
    #
    # For the 4x2x2 block: schoolbook rank = 16
    # For the 4x2x3 block: schoolbook rank = 24
    # For the 4x3x2 block: schoolbook rank = 24
    # For the 4x3x3 block: schoolbook rank = 36
    # Total = 16 + 24 + 24 + 36 = 100, which is worse than schoolbook 100.
    #
    # Wait, schoolbook for 4x5x5 is 100. So block decomposition with schoolbook doesn't help.
    # We need to use faster algorithms for the sub-blocks.
    #
    # For 2x2 matrix multiplication: Strassen gives rank 7 (vs schoolbook 8)
    # For 3x3 matrix multiplication: Laderman gives rank 23 (vs schoolbook 27)
    #
    # But our blocks are 4x2x2, 4x2x3, 4x3x2, 4x3x3, not square.
    # Let me reconsider.
    #
    # The tensor for multiplying A (n x m) by B (m x p) to get C (n x p) has:
    #   U: (n*m) x rank
    #   V: (m*p) x rank
    #   W: (p*n) x rank
    #
    # For a block where m is split into m1, m2 and p into p1, p2:
    #   Block (m1, p1): n x m1 x p1, rank = n*m1*p1 (schoolbook)
    #   etc.
    #
    # Actually, I think the key insight is that for the sub-blocks, we can use known optimal
    # decompositions. Let me try a different approach.
    #
    # Let's just do a careful schoolbook decomposition and verify it works.
    # The rank is n*m*p = 100.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
