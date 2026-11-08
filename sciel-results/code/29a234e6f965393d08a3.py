"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will try to find a rank-15 decomposition for 2x3x3 matrix multiplication.
    # The tensor is T[i,j,k] = delta[i,k] (where i in [0,n), j in [0,m), k in [0,p))
    # We seek U, V, W such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known result: rank(2x3x3) = 15.
    # Let's construct it using a block approach or known explicit formulas.
    # One standard way is to view it as 2x2x2 (rank 7) plus extra terms.
    # But 2x3x3 is not just 2x2x2.
    
    # Let's try to construct a rank 15 decomposition explicitly.
    # U shape: (6, 15)
    # V shape: (9, 15)
    # W shape: (6, 15)
    
    # Indices:
    # U[i, j] for i in 0..1 (rows of A), j in 0..2 (cols of A)
    # V[j, k] for j in 0..2 (cols of A), k in 0..2 (rows of B)
    # W[k, i] for k in 0..2 (rows of B), i in 0..1 (cols of C)
    
    # Note: The problem statement says:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    
    # Let's use a known rank 15 decomposition for 2x3x3.
    # Reference: "The rank of 2x3x3 matrix multiplication is 15"
    # We can build it by combining:
    # 1. A 2x2x2 Strassen decomposition (7 terms) for the top-left 2x2 block of the result?
    #    No, the matrices are 2x3 and 3x3.
    #    A = [[a00, a01, a02],
    #         [a10, a11, a12]]
    #    B = [[b00, b01, b02],
    #         [b10, b11, b12],
    #         [b20, b21, b22]]
    #    C = A*B = [[c00, c01, c02],
    #               [c10, c11, c12]]
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # We can group terms.
    # Consider the first two columns of B: B1 = B[:, 0:2] (3x2)
    # And the last column: B2 = B[:, 2] (3x1)
    # C[:, 0:2] = A * B1 (2x3 * 3x2 -> 2x2)
    # C[:, 2] = A * B2 (2x3 * 3x1 -> 2x1)
    
    # Rank of 2x3x2 matrix multiplication?
    # It is equivalent to 2x2x2 with an extra column in A.
    # Actually, 2x3x2 rank is 12? No.
    # Let's look up or derive.
    # Rank of 2x2x2 is 7.
    # Rank of 2x3x3 is 15.
    
    # Let's try to construct it manually.
    # We can use the fact that we can compute A*B by:
    # 1. Compute M = A * B_sub where B_sub is 3x2 (first two cols of B).
    #    This is a 2x3x2 multiplication.
    # 2. Compute N = A * B_col where B_col is 3x1 (last col of B).
    #    This is a 2x3x1 multiplication.
    
    # What is the rank of 2x3x2?
    # It can be seen as two 2x3x1 multiplications? No.
    # Actually, 2x3x2 is isomorphic to 2x2x2 with padding?
    # Let's assume we can do 2x3x2 in rank 12? Or 11?
    # If rank(2x3x2) = 12 and rank(2x3x1) = 6, total 18.
    # But we need 15.
    
    # Let's try a specific decomposition.
    # U, V, W as lists of lists of Fractions.
    
    # I will use a known rank 15 decomposition from literature or derive it.
    # One construction:
    # Use Strassen-like ideas.
    
    # Let's define the 15 terms.
    # Term 1-7: Strassen for the 2x2 part of B (cols 0,1) and A (rows 0,1)?
    # No, A is 2x3.
    
    # Let's try to find a decomposition online or in memory.
    # "Matrix multiplication 2x3x3 rank 15"
    # A known decomposition uses 15 multiplications.
    
    # Here is a candidate decomposition:
    # We can express C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's use the following structure:
    # We will define U, V, W as numpy arrays of Fractions.
    
    # Since I don't have the exact 15-term decomposition memorized, I will try to construct one
    # by solving the system of equations for rank 15, or use a known one.
    
    # Actually, I can use a rank 18 schoolbook decomposition and try to optimize it,
    # but the task requires exact decomposition.
    
    # Let's use a known rank 15 decomposition for 2x3x3.
    # Reference: "On the rank of 2x3x3 matrix multiplication"
    
    # I will use the following explicit decomposition:
    # U (6x15), V (9x15), W (6x15)
    
    # Let's try to build it from 2x2x2 (7 terms) and 2x3x1 (6 terms) and something else?
    # 7 + 6 = 13. Need 2 more.
    
    # Actually, rank(2x3x3) = 15 is a known result.
    # Let's use a decomposition that combines:
    # - 7 terms for 2x2x2 (using first 2 cols of B and both rows of A)
    # - 6 terms for 2x3x1 (using last col of B and both rows of A)
    # - 2 terms for the interaction?
    
    # This is tricky. Let's try to use a generic rank minimization approach with a time limit.
    # But the task asks for a static program.
    
    # I will use a known rank 15 decomposition if I can find it.
    # If not, I will use rank 18 (schoolbook) which is safe.
    # But the goal is to minimize rank.
    
    # Let's try to construct rank 15.
    # One possible decomposition:
    # U = [
    #   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    # This is just the identity for U, which means we are doing schoolbook.
    
    # I will use the rank 18 schoolbook decomposition for now, as it is guaranteed correct.
    # If I can find a rank 15 one, I will use it.
    
    # Let's try to use a rank 15 decomposition from a known source.
    # "The rank of 2x3x3 matrix multiplication is 15"
    # One decomposition is:
    # U = np.array([...])
    # V = np.array([...])
    # W = np.array([...])
    
    # Since I cannot recall the exact 15-term decomposition, I will use rank 18.
    # But wait, the task says "Find a rank-minimal exact tensor decomposition".
    # The minimal rank is 15.
    
    # Let's try to construct it.
    # We can use the fact that 2x3x3 can be decomposed into:
    # 1. 2x2x2 (7 terms) for C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2]
    # 2. 2x1x2 (4 terms?) for C[0:2, 2] using A[0:2, 0:2] and B[0:2, 2]
    # 3. 1x3x2 (3 terms?) for C[2, 0:2] ... no, C is 2x3.
    
    # This is getting complicated. I will use rank 18.
    
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
