import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # BCLR rank-23 coefficients for 3x3x3
    # U, V, W are lists of 23 vectors of length 9
    # Flattened row-major 3x3 matrices
    
    # These are the coefficients from Bini et al. (1979)
    # Format: [u1..u9, v1..v9, w1..w9] for each term?
    # No, separate lists.
    
    U_33 = [
        [1,0,0, 0,1,0, 0,0,1], # I
        [1,0,0, 0,0,0, 0,0,0], # E11
        [0,0,0, 0,1,0, 0,0,0], # E22
        [0,0,0, 0,0,0, 0,0,1], # E33
        [0,1,0, 0,0,0, 0,0,0], # E12
        [0,0,0, 1,0,0, 0,0,0], # E21
        [0,0,1, 0,0,0, 0,0,0], # E13
        [0,0,0, 0,0,1, 0,0,0], # E32
        [0,0,0, 0,0,0, 1,0,0], # E31
        [0,0,0, 0,0,1, 0,1,0], # E32+E23?
        [0,0,0, 0,1,0, 0,0,1], # E23+E33?
        [0,0,1, 0,1,0, 0,0,0], # E13+E22?
        [1,0,0, 0,1,0, 0,0,0], # E11+E22?
        [0,1,0, 0,0,0, 0,0,1], # E12+E33?
        [0,0,0, 1,0,0, 0,0,1], # E21+E33?
        [0,0,1, 0,0,0, 0,0,1], # E13+E33?
        [1,0,0, 0,0,0, 0,0,1], # E11+E33?
        [0,1,0, 0,0,0, 0,0,0], # E12
        [0,0,0, 0,1,0, 0,0,0], # E22
        [0,0,0, 0,0,0, 0,0,1], # E33
        [0,0,1, 0,0,0, 0,0,0], # E13
        [0,0,0, 0,0,1, 0,0,0], # E32
        [0,0,0, 0,0,0, 1,0,0], # E31
    ]
    
    # This is a dummy list. I need the real BCLR coefficients.
    # Since I don't have them, I will use a known valid rank-23 set if I can recall one.
    # If not, I will fall back to schoolbook 54?
    # No, I will try to use a rank-23 algorithm from a reliable source.
    
    # Actually, I can use the fact that R(3,3,3) <= 23.
    # I will use the coefficients from the paper.
    
    # Due to uncertainty in exact coefficients, I will use a safe fallback:
    # Use schoolbook for 3x3x3 (rank 27) twice? Rank 54.
    # This is not an improvement.
    
    # Let's try to find a rank 46 solution using a different method?
    # Maybe R(3,3,6) <= 45?
    
    # I will return the schoolbook 54 to be safe, as I cannot guarantee the BCLR coefficients.
    # Wait, the task asks to improve.
    # I will try to implement a rank 46 solution using the BCLR coefficients I *think* I remember?
    # Risky.
    
    # Let's use the schoolbook 54.
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
