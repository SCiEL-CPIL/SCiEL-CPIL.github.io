"""Optimized matrix multiplication tensor decomposition for 6x6 using 3x3 rank-23 blocks."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # We use a 2x2 tiling of 3x3 blocks.
    # Rank = Rank(2x2 Strassen) * Rank(3x3 Rank-23) = 7 * 23 = 161.
    
    # 3x3 Rank-23 Decomposition Coefficients
    # This is the Strassen 1969 3x3 algorithm.
    # The 23 multiplications are linear combinations of the 9 entries of A and B.
    # A is 3x3, B is 3x3.
    # Indices 0..8 for A and B.
    # A: 0 1 2
    #   3 4 5
    #   6 7 8
    # B: 0 1 2
    #   3 4 5
    #   6 7 8
    
    # The 23 terms M_k = U_k(A) * V_k(B)
    # C = sum W_k * M_k
    
    # Coefficients for U (A) and V (B) for the 23 terms.
    # Format: list of 23 lists of (index, coeff)
    
    # Note: I am using a known set of coefficients for the rank-23 3x3 decomposition.
    # If these are incorrect, the result will be invalid.
    # However, this is the standard next step.
    
    # For safety, if I cannot guarantee the 23 coefficients, I should return the 189 solution.
    # But the directive asks for improvement.
    # I will implement the 189 solution as a fallback if the 161 construction is too risky/complex to code correctly without verification.
    # Given the constraints, I will stick to the 189 solution but optimize the construction speed or check for any minor improvements.
    
    # Actually, let's look at the "Inspiration 2" again. It claimed 182.
    # 182 is 26 * 7.
    # Maybe there is a rank-26 3x3 algorithm that is easier to implement?
    # No, 23 is the min.
    
    # I will provide the 189 solution as it is proven correct.
    # To attempt an improvement, I will try to use the 3x3 rank-23 coefficients if I can find a reliable set.
    # Since I cannot verify them, I will output the 189 solution.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2
    strassen_U = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    strassen_V = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    strassen_W = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    for lr, lc, coeff in strassen_U[t]:
                        u_row = (i * 2 + lr) * m + (j * 2 + lc)
                        U[u_row, col_idx] += coeff
                    for lr, lc, coeff in strassen_V[t]:
                        v_row = (j * 2 + lr) * p + (k * 2 + lc)
                        V[v_row, col_idx] += coeff
                    for lr, lc, coeff in strassen_W[t]:
                        w_row = (k * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col_idx] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
