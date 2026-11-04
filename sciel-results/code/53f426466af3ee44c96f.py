import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 Winograd F(4) rank 12
    # U1, V1, W1 are 16x12
    # Part 2: 4x4x1 Schoolbook rank 16
    # U2, V2, W2 are 16x16, 4x16, 4x16
    
    # Since I don't have the exact Winograd matrices hardcoded in my head with 100% confidence 
    # to ensure exactness without testing, and the prompt requires EXACT decomposition, 
    # I will use the Schoolbook for the 4x4x4 part as well? 
    # No, that's rank 64 + 16 = 80.
    
    # Let's try to find a safe low rank.
    # The rank of 4x4x4 is 12.
    # I will use the known Winograd matrices.
    
    # U_win (16x12)
    # V_win (16x12)
    # W_win (16x12)
    
    # I will define them as lists.
    # If I fail, I return schoolbook.
    
    # For safety, I will implement the Schoolbook 80 if I cannot guarantee Winograd.
    # But the directive wants improvement.
    
    # I will use the Winograd matrices from a standard source.
    # Reference: https://en.wikipedia.org/wiki/Winograd_algorithm
    # The 4x4 case is not fully detailed in the wiki snippet I recall, but it is standard.
    
    # I will use the rank 28 construction.
    
    # U_win
    u_win = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # A00
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], # A01
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], # A02
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1], # A03
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], # A10
        [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1], # A11
        [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1], # A12
        [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1], # A13
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], # A20
        [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1], # A21
        [1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1], # A22
        [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1], # A23
        [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1], # A30
        [1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1], # A31
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1], # A32
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1]  # A33
    ]
    # This is a guess based on Hadamard-like structures. It might not be correct for Winograd.
    # Winograd uses specific coefficients.
    
    # Given the risk of incorrect Winograd matrices, I will stick to the Schoolbook 80 
    # but optimize the code structure? No, score is rank.
    
    # I will return the Schoolbook decomposition.
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
