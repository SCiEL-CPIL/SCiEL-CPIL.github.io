"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 4 blocks of (Strassen 2x2 + Strassen 2x2)
    # Each block handles 2 output columns.
    # Block r corresponds to output columns 2r, 2r+1.
    # It uses A[:, 0:2] * B[0:2, 2r:2r+2] + A[:, 2:4] * B[2:4, 2r:2r+2]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 multiplication
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Output reconstruction:
    # z11 = m1 + m4 - m5 + m7
    # z12 = m3 + m5
    # z21 = m2 + m4
    # z22 = m1 - m2 + m3 + m6
    
    # We will define the terms for a single Strassen 2x2 multiplication
    # Inputs: a,b,c,d for X; e,f,g,h for Y
    # Terms:
    # 1: (a+d)(e+h) -> X: [1,0,1,0], Y: [1,1,1,1]
    # 2: (c+d)e     -> X: [0,0,1,1], Y: [1,0,0,0]
    # 3: a(f-h)     -> X: [1,0,0,0], Y: [0,1,0,-1]
    # 4: d(g-c)     -> X: [0,0,-1,1], Y: [0,0,1,0]
    # 5: (a+b)h     -> X: [1,1,0,0], Y: [0,0,0,1]
    # 6: (c-a)(e+f) -> X: [-1,-1,1,0], Y: [1,1,0,0]
    # 7: (b-d)(g+h) -> X: [0,1,0,-1], Y: [0,0,1,1]
    
    # Note: The input vectors for X and Y are flattened.
    # X = [[a, b], [c, d]] -> [a, b, c, d]
    # Y = [[e, f], [g, h]] -> [e, f, g, h]
    
    strassen_terms = [
        ([1, 0, 1, 0], [1, 1, 1, 1], [1, 0, 0, 1]),  # m1 -> z11, z22
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # m2 -> z21
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3 -> z12, z22
        ([0, 0, -1, 1], [0, 0, 1, 0], [1, 0, 1, 0]),  # m4 -> z11, z21
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5 -> z11, z12
        ([-1, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # m6 -> z22
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7 -> z11
    ]
    # Correction on W coefficients (z11, z12, z21, z22)
    # z11 = m1 + m4 - m5 + m7
    # z12 = m3 + m5
    # z21 = m2 + m4
    # z22 = m1 - m2 + m3 + m6
    
    # Let's re-verify the W coefficients for each term index 0..6
    # Term 0 (m1): z11=1, z22=1. W=[1,0,0,1]
    # Term 1 (m2): z21=1. W=[0,0,1,0]
    # Term 2 (m3): z12=1, z22=1. W=[0,1,0,1]
    # Term 3 (m4): z11=1, z21=1. W=[1,0,1,0]
    # Term 4 (m5): z11=-1, z12=1. W=[-1,1,0,0]
    # Term 5 (m6): z22=1. W=[0,0,0,1]
    # Term 6 (m7): z11=1. W=[1,0,0,0]
    
    corrected_w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for r in range(4):
        # Block r:
        # Output cols: 2r, 2r+1
        # A_L: cols 0,1. A_R: cols 2,3.
        # B_r: rows 0,1, cols 2r, 2r+1.
        # B'_r: rows 2,3, cols 2r, 2r+1.
        
        # First Strassen: A_L * B_r
        # X = A_L (2x2), Y = B_r (2x2)
        # X indices in A: (0,0), (0,1), (1,0), (1,1) -> U rows: 0, 1, 4, 5
        # Y indices in B: (0,2r), (0,2r+1), (1,2r), (1,2r+1) -> V rows: 0*8+2r, 0*8+2r+1, 1*8+2r, 1*8+2r+1
        
        for t in range(7):
            ux, uy, wz = strassen_terms[t]
            # Fill U for X
            # X = [a, b, c, d]
            # a: U[0, col] += ux[0]
            # b: U[1, col] += ux[1]
            # c: U[4, col] += ux[2]
            # d: U[5, col] += ux[3]
            U[0][col] += ux[0]
            U[1][col] += ux[1]
            U[4][col] += ux[2]
            U[5][col] += ux[3]
            
            # Fill V for Y
            # Y = [e, f, g, h]
            # e: V[0*8 + 2r, col] += uy[0]
            # f: V[0*8 + 2r+1, col] += uy[1]
            # g: V[1*8 + 2r, col] += uy[2]
            # h: V[1*8 + 2r+1, col] += uy[3]
            V[0 * 8 + 2 * r][col] += uy[0]
            V[0 * 8 + 2 * r + 1][col] += uy[1]
            V[1 * 8 + 2 * r][col] += uy[2]
            V[1 * 8 + 2 * r + 1][col] += uy[3]
            
            # Fill W for Z
            # Z = [z11, z12, z21, z22]
            # z11: W[2r*2 + 0, col] += wz[0]
            # z12: W[2r*2 + 1, col] += wz[1]
            # z21: W[2r*2 + 0 + 2, col] += wz[2] -> Wait, W index is k*n + i.
            # k is col of C, i is row of C.
            # C is 2x8.
            # Block r corresponds to C cols 2r, 2r+1.
            # z11 is C[0, 2r]. W index: (2r)*2 + 0 = 4r.
            # z12 is C[0, 2r+1]. W index: (2r+1)*2 + 0 = 4r+2.
            # z21 is C[1, 2r]. W index: (2r)*2 + 1 = 4r+1.
            # z22 is C[1, 2r+1]. W index: (2r+1)*2 + 1 = 4r+3.
            
            W[4 * r][col] += wz[0]
            W[4 * r + 2][col] += wz[1]
            W[4 * r + 1][col] += wz[2]
            W[4 * r + 3][col] += wz[3]
            
            col += 1
        
        # Second Strassen: A_R * B'_r
        # X = A_R (2x2), Y = B'_r (2x2)
        # X indices in A: (0,2), (0,3), (1,2), (1,3) -> U rows: 2, 3, 6, 7
        # Y indices in B: (2,2r), (2,2r+1), (3,2r), (3,2r+1) -> V rows: 2*8+2r, 2*8+2r+1, 3*8+2r, 3*8+2r+1
        
        for t in range(7):
            ux, uy, wz = strassen_terms[t]
            # Fill U for X
            U[2][col] += ux[0]
            U[3][col] += ux[1]
            U[6][col] += ux[2]
            U[7][col] += ux[3]
            
            # Fill V for Y
            V[2 * 8 + 2 * r][col] += uy[0]
            V[2 * 8 + 2 * r + 1][col] += uy[1]
            V[3 * 8 + 2 * r][col] += uy[2]
            V[3 * 8 + 2 * r + 1][col] += uy[3]
            
            # Fill W for Z (same output block)
            W[4 * r][col] += wz[0]
            W[4 * r + 2][col] += wz[1]
            W[4 * r + 1][col] += wz[2]
            W[4 * r + 3][col] += wz[3]
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
