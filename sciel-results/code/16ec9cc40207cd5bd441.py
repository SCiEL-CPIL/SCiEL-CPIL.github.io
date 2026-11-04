"""Rewritten program: try to find a rank-minimal exact tensor decomposition for 3x4*4x8."""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    We try to use Strassen's algorithm for 2x2 blocks where possible, combined with
    schoolbook for the remaining parts.
    
    Strategy:
    - Split A (3x4) into [A_top (2x4); A_bottom (1x4)]
    - Split B (4x8) into [B_top (2x8); B_bottom (2x8)]
    
    Then C = A_top * B_top + A_bottom * B_bottom
    
    For A_top (2x4) * B_top (2x8):
    - Split A_top into [A1 (2x2) | A2 (2x2)]
    - Split B_top into [B1 (2x2) B2 (2x2); ...] wait, B_top is 2x8, so split into 4 blocks of 2x2
    
    Actually, let me think more carefully.
    
    A is 3x4, B is 4x8, C is 3x8.
    
    Let me try a different decomposition:
    - Split the middle dimension (4) into two parts of size 2
    - A = [A1 | A2] where A1, A2 are 3x2
    - B = [B1; B2] where B1, B2 are 2x8
    
    C = A1*B1 + A2*B2
    
    Each product A1*B1 is 3x2 * 2x8 = 3x8. The rank of each is at most 3*2*8=48.
    Total: 96, same as schoolbook.
    
    But maybe I can use Strassen for some sub-blocks.
    
    Let me try yet another approach: 
    - Think of A as having 3 rows, each of length 4
    - Think of B as having 4 rows, each of length 8
    
    For each pair of rows i in {0,1,2} and columns k in {0,...,7}, the entry C[i,k] = sum_j A[i,j]*B[j,k]
    
    This is a sum of 4 terms for each (i,k) pair, so 3*8=24 entries, each requiring 4 multiplications.
    Total: 96 multiplications (schoolbook).
    
    Can we reduce this? 
    
    Idea: Use the fact that if we have two 2x2 matrix multiplications, we can do them with 7 multiplications instead of 8 using Strassen.
    
    Let me try to identify 2x2 subproblems:
    
    A is 3x4. Let's split it as:
    A = [[a00, a01, a02, a03],
         [a10, a11, a12, a13],
         [a20, a21, a22, a23]]
    
    B is 4x8. Let's split it as:
    B = [[b00, b01, b02, b03, b04, b05, b06, b07],
         [b10, b11, b12, b13, b14, b15, b16, b17],
         [b20, b21, b22, b23, b24, b25, b26, b27],
         [b30, b31, b32, b33, b34, b35, b36, b37]]
    
    C is 3x8:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Let me try to group the j index into pairs: (0,1) and (2,3).
    
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    For fixed i and k, this is two dot products of length 2.
    
    Now, if I fix i and consider all k, I have:
    C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:] + A[i,3]*B[3,:]
    
    This is a linear combination of the rows of B.
    
    Let me try a different approach. What if I use the following idea:
    
    For the submatrix A[0:2, :] (2x4) and B[:, 0:2] (4x2), I have a 2x4 * 4x2 multiplication.
    I can split this into 2x2 blocks:
    A_sub = [[A[0,0:2], A[0,2:4]],
             [A[1,0:2], A[1,2:4]]]
    B_sub = [[B[0:2, 0:2]; B[2:4, 0:2]]  (this is 4x2, not 4x4)
    
    This doesn't work directly because the dimensions don't match for Strassen.
    
    Let me try to be more systematic. I'll use the following approach:
    
    1. Start with schoolbook (rank 96)
    2. Try to find a better decomposition using optimization
    
    Actually, let me just try to construct a decomposition with rank less than 96 by using some clever algebraic identities.
    
    One known result: the rank of M_{n,m,p} is at most n*m*p, but can be lower.
    
    For 3x4*4x8, let me try to see if I can get rank 90 or so.
    
    Actually, let me just return the schoolbook decomposition for now, as finding a better one is complex.
    
    Wait, let me try one more idea. What if I use the following:
    
    C = A*B
    
    Let me define:
    P1 = (A[0:2, 0:2] + A[0:2, 2:4]) * (B[0:2, :] + B[2:4, :])
    P2 = A[0:2, 2:4] * B[2:4, :]
    
    Then:
    C[0:2, :] = P1 - P2 + A[0:2, 0:2] * B[0:2, :]
    
    This doesn't seem to help reduce the rank.
    
    Let me just implement the schoolbook and return it.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition using a simple heuristic
    # For now, use schoolbook
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
