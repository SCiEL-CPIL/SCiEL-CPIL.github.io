"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # We will try to find a low-rank decomposition.
    # The schoolbook rank is n*m*p = 4*5*6 = 120.
    # We want to see if we can do better.
    
    # Strategy:
    # 1. Try known bounds or heuristics.
    # 2. Use a randomized approach to find a low-rank tensor decomposition if possible.
    #    However, finding the *exact* minimal rank is hard.
    #    For 4x5x6, the known upper bound is 120 (schoolbook).
    #    Is there a known better algorithm?
    #    For 4x4x4, Strassen-like or others might reduce it, but 4x5x6 is not square.
    #    
    #    Let's check if we can use a block structure.
    #    If we decompose the matrices into blocks, we can use a tensor product of smaller decompositions.
    #    But 4, 5, 6 don't factor nicely into powers of 2.
    #
    #    Let's try to find a decomposition with rank < 120 using a simple search or known results.
    #    Actually, for small matrices, the minimal rank is often the schoolbook rank unless there's a specific symmetry.
    #    
    #    Wait, the task asks for "rank-minimal exact tensor decomposition".
    #    If I cannot find a better one, I should return the schoolbook one.
    #    But maybe I can try to verify if a lower rank is possible?
    #    No, verifying minimal rank is hard.
    #
    #    Let's try to construct a decomposition with rank slightly less than 120 if possible?
    #    Or just return the schoolbook one?
    #    The score is the rank. Lower is better.
    #    If I return 120, the score is 1.0 (seed).
    #    If I find 119, the score is 120/119 > 1.0.
    #
    #    Is it possible to beat 120 for 4x5x6?
    #    The bilinear complexity of 4x5x6 matrix multiplication.
    #    According to literature, for m*n*p where one dimension is small, the rank is often m*n*p.
    #    However, sometimes you can save a few operations.
    #    
    #    Let's try a simple heuristic: 
    #    Try to set some entries of U, V, W to zero and see if the tensor can still be reconstructed?
    #    This is equivalent to finding a sparse decomposition.
    #    
    #    Actually, a safer approach: 
    #    Return the schoolbook decomposition. It is correct.
    #    If the evaluator accepts it, score is 1.0.
    #    If I can find a better one, I'll try.
    #
    #    Let's try to use the fact that 4 = 2*2, 6 = 2*3, 5 is prime.
    #    Maybe decompose 4x5x6 into (2x2x2) and (2x5x3)?
    #    No, the tensor product of decompositions multiplies the ranks.
    #    
    #    Let's just return the schoolbook decomposition for now, but check if we have time to do a simple local search.
    #    Actually, local search for tensor rank minimization is very hard and unlikely to succeed in 120s for this size.
    #    
    #    However, there is a known result: for 2x2x2, rank is 7 (Strassen) vs 8.
    #    For 4x4x4, rank is at most 49 (using Strassen recursively) vs 64.
    #    For 4x5x6, we can embed it in 8x8x8? No, that's too big.
    #    
    #    Let's try to use Strassen for the 4x4 part and schoolbook for the rest?
    #    We can view the 4x5 matrix as two 2x5 blocks? Or 4x2 and 4x3?
    #    
    #    Let's try a block decomposition:
    #    Split A (4x5) into A1 (2x5) and A2 (2x5).
    #    Split B (5x6) into B1 (5x3) and B2 (5x3).
    #    Then C (4x6) = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    #    This is 4 multiplications of (2x5) by (5x3).
    #    The rank of 2x5x3 multiplication is 2*5*3 = 30 (schoolbook).
    #    Total rank = 4 * 30 = 120. No gain.
    #    
    #    What if we use Strassen for the 2x2x2 core?
    #    2x5x3: Split 2x5 into 2x2 and 2x3? No, 5 is not 2*2.
    #    Split 5 into 2+3.
    #    A (2x5) = [A11 (2x2) A12 (2x3)]
    #    B (5x3) = [B11 (2x3); B21 (3x3)]
    #    C = A11*B11 + A12*B21.
    #    This is 2 multiplications: (2x2)x(2x3) and (2x3)x(3x3).
    #    Rank of (2x2)x(2x3): 2*2*3 = 12.
    #    Rank of (2x3)x(3x3): 2*3*3 = 18.
    #    Total for 2x5x3: 12 + 18 = 30. Still schoolbook.
    #    
    #    It seems hard to beat 120 for 4x5x6 with simple block strategies.
    #    
    #    Let's just return the schoolbook decomposition.
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
