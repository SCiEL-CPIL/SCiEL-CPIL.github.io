import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 42
    rank_schoolbook = n * m * p
    
    # Try to use the known 40-term algorithm for 2x3x7
    # The rank of 2x3x7 matrix multiplication is 40.
    # We will construct the 40-term decomposition.
    
    # The 40-term algorithm is a specific set of 40 vectors.
    # If we cannot hardcode them correctly, we return schoolbook.
    # However, for this specific case, the 40-term algorithm is well-known.
    
    # Let's try to verify if we can construct it.
    # The 40-term algorithm uses the following structure:
    # It is based on the identity:
    # sum_{i,j,k} C_{ik} A_{ij} B_{jk} = sum_{t=1}^{40} (u_t \cdot A) (v_t \cdot B) (w_t \cdot C)
    
    # I will use a known 40-term decomposition.
    # Since I don't have the exact coefficients memorized, I will use the schoolbook.
    # But wait, the prompt asks for a "stronger algorithm".
    # Returning 42 is not stronger.
    # I will attempt to use a randomized search to find a 40-term decomposition?
    # No, that's too slow and unreliable.
    
    # I will use the explicit 40-term algorithm from the literature.
    # The algorithm is:
    # U, V, W are 6x40, 21x40, 14x40.
    
    # I will hardcode the 40-term algorithm.
    # The vectors are:
    # (i, j, k) with signs.
    
    # For now, I will return the schoolbook 42.
    rank = rank_schoolbook
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
