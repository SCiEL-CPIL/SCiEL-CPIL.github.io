"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Exploits block structure: B (3x9) = [B0 | B1 | B2] where each Bb is 3x3.
    For each row i of A (i=0,1) and each block b (b=0,1,2), compute A[i,:] @ Bb.
    This is a 1x3 * 3x3 multiplication, decomposed with rank 9 (trivial per-output).
    Total rank = 2 * 3 * 9 = 54... wait, that's the same.
    
    Let me reconsider. For 1x3 * 3x3, each of the 3 outputs is:
    out_j = v0*M[0,j] + v1*M[1,j] + v2*M[2,j]
    
    I can use rank 3 per output if I treat each output independently, but they share v.
    Actually, the minimum rank for 1x3*3x3 is 3 (achieved by:
    - p0 = v0 * (M[0,0], M[0,1], M[0,2])  -- but this is 3 multiplications
    
    Let me just use the straightforward rank-9 per block:
    For output j in {0,1,2} of block b:
      out_j = sum_{i=0}^{2} v_i * M[i, j]
    This gives 3*3 = 9 rank-1 terms per block.
    
    Total: 2 rows * 3 blocks * 9 = 54. Same as schoolbook!
    
    I need a better approach. Let me think about sharing across blocks.
    
    Actually, the key insight: A is 2x3, so for row i, A[i,:] = (a_i0, a_i1, a_i2).
    B is 3x9. The product C[i,k] = sum_j A[i,j] * B[j,k] for k=0..8.
    
    For fixed i, this is a 1x3 vector times a 3x9 matrix = 1x9 vector.
    The rank of 1x3 * 3x9 multiplication: each of the 9 outputs is a bilinear form.
    Minimum rank is at most 3*9=27, but can be less due to shared structure.
    
    Actually, I recall that for 1xm * mxn, the rank is mn (no improvement possible in general).
    
    Let me try a different decomposition. What if I use Strassen-like ideas on the 3x3 blocks?
    
    For 3x3 * 3x3, Strassen gives rank 7 instead of 27.
    But here I have 1x3 * 3x3, which is different.
    
    Let me just implement the rank-54 schoolbook but verify it's correct, and see if I can find
    any known lower-rank algorithms for this specific case.
    
    For now, return the correct rank-54 decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
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
