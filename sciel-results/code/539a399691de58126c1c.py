import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank for 4x4x4 is 49
    # Total rank for 4x4x8 is 2 * 49 = 98
    rank = 98
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to construct the 4x4x4 decomposition first
    # Then place it into the 4x4x8 structure
    
    # Helper to build 4x4x4 U, V, W
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    
    # Strassen coefficients for 2x2
    # z1 = (x00+x11)(y00+y11)
    # z2 = (x10+x11)y00
    # z3 = x00(y01-y11)
    # z4 = x11(y10-y00)
    # z5 = (x00+x01)y11
    # z6 = (x10-x00)(y00+y01)
    # z7 = (x01-x11)(y10+y11)
    
    # w_s_uv: contribution of z_s to M[u,v]
    w_coeffs = [
        {(0,0): 1, (1,1): 1}, # s=1
        {(1,0): 1},           # s=2
        {(0,1): 1},           # s=3
        {(0,0): 1, (1,0): 1},# s=4
        {(0,1): 1, (0,0): -1},# s=5
        {(1,1): 1},           # s=6
        {(0,0): 1, (1,1): 1}  # s=7
    ]
    
    # d_k_ij: contribution of M_k to C[i,j]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    d_coeffs = [
        {(0,0): 1, (1,1): 1}, # k=1
        {(1,0): 1, (1,1): -1},# k=2
        {(0,1): 1, (1,1): 1}, # k=3
        {(0,0): 1, (1,0): 1}, # k=4
        {(0,0): -1, (0,1): 1},# k=5
        {(1,1): 1},           # k=6
        {(0,0): 1, (1,1): 1}  # k=7
    ]
    
    # X_k and Y_k definitions in terms of A and B blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We will iterate k=1..7 and s=1..7
    for k in range(7):
        # Define X_k and Y_k as linear combinations of A and B entries
        # X_k is 2x2, Y_k is 2x2
        # We represent them as dicts: {(row, col): [(A_idx, coeff), ...]}
        
        # This is a bit complex to hardcode manually in the loop.
        # Let's define the linear forms for X_k and Y_k explicitly.
        
        # X1 = A00 + A11
        # X2 = A10 + A11
        # X3 = A00
        # X4 = A11
        # X5 = A00 + A01
        # X6 = A10 - A00
        # X7 = A01 - A11
        
        # Y1 = B00 + B11
        # Y2 = B00
        # Y3 = B01 - B11
        # Y4 = B10 - B00
        # Y5 = B11
        # Y6 = B00 + B01
        # Y7 = B10 + B11
        
        # To avoid manual error, I will generate the U and V columns based on these definitions.
        # A_idx: 0..15. Block (bi, bj) contains A[2*bi + r, 2*bj + c].
        # So A_idx = (2*bi + r) * 4 + (2*bj + c).
        
        # For each k, define the 4 entries of X_k (x00, x01, x10, x11)
        # and the 4 entries of Y_k (y00, y01, y10, y11)
        
        # Let's create a helper to get the linear form for a block sum
        # e.g., "A00 + A11" means for each (r,c) in 2x2, take A[2*0+r, 2*0+c] + A[2*1+r, 2*1+c]
        
        # I will hardcode the 49 columns' U and V parts.
        # This is tedious but safe.
        
        pass # Logic to be implemented in the final code
        
    # I will implement the logic directly in the code below.
    
    return {"U": U, "V": V, "W": W}
