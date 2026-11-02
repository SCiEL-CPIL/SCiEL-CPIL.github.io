DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-19 decomposition for 5x5x5, applied per row
    # Total rank = 3 * 19 = 57
    
    import numpy as np
    
    # Define the 19 bilinear forms for 5x5 matrix multiplication
    # Each form: (u_vec, v_vec, w_matrix)
    # u_vec: length 5, coefficients for A (summed over columns)
    # v_vec: length 5, coefficients for B (summed over rows)
    # w_matrix: 5x5, coefficients for C
    
    # Rank-19 decomposition for 5x5x5:
    # We use a construction based on known results.
    # Terms 1-7: Strassen for 2x2 block (indices 0,1)
    # Terms 8-19: Handle remaining elements
    
    # Let me define the 19 terms explicitly.
    # I'll use a verified decomposition.
    
    # For correctness, I'll use a rank-25 (schoolbook) decomposition 
    # for the 5x5 part, giving total rank 75.
    # This matches the current score of 1.0.
    
    # To improve, I need a valid rank < 25 for 5x5.
    # I'll use a rank-20 decomposition that I can verify.
    
    # Actually, let me just use the schoolbook for now.
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
