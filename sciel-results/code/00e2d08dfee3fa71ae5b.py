import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A1 (2x2) * B1 (2x8) using Block Strassen
    # A1 indices: (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,2,3
    # B1 indices: (0,0)..(0,7), (1,0)..(1,7) -> V indices 0..7, 8..15
    # C indices: (0,0)..(0,7), (1,0)..(1,7) -> W indices 0..7, 8..15
    
    # Strassen coefficients for A (2x2)
    # M1: (A00 + A11)
    # M2: (A10 + A11)
    # M3: A00
    # M4: A11
    # M5: (A00 + A01)
    # M6: (A10 - A00)
    # M7: (A01 - A11)
    
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Strassen coefficients for B block (2x2)
    # M1: (B00 + B11)
    # M2: B00
    # M3: (B10 - B11)
    # M4: (B00 - B01)
    # M5: B11
    # M6: (B00 + B01)
    # M7: (B10 + B11)
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [1, -1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction coefficients for C (2x2)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    col = 0
    for k in range(4): # 4 blocks of B1
        for l in range(7): # 7 Strassen steps
            # U: A coefficients
            for ai in range(4):
                U[ai, col] = A_coeffs[l][ai]
                
            # V: B coefficients for block k
            # B block k uses B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
            # V indices:
            # B[0, 2k] is at index 0 + 2k
            # B[0, 2k+1] is at index 0 + 2k + 1
            # B[1, 2k] is at index 8 + 2k
            # B[1, 2k+1] is at index 8 + 2k + 1
            v_indices = [2*k, 2*k+1, 8+2*k, 8+2*k+1]
            for bi in range(4):
                V[v_indices[bi], col] = B_coeffs[l][bi]
                
            # W: C reconstruction
            # C[0, 2k] and C[0, 2k+1] come from C00, C01 of the block
            # C[1, 2k] and C[1, 2k+1] come from C10, C11 of the block
            # W indices:
            # C[0, 2k] is at 0 + 2k
            # C[0, 2k+1] is at 0 + 2k + 1
            # C[1, 2k] is at 8 + 2k
            # C[1, 2k+1] is at 8 + 2k + 1
            
            # C00 coeff is C_coeffs[0][l]
            W[2*k, col] = C_coeffs[0][l]
            W[2*k+1, col] = C_coeffs[1][l]
            W[8+2*k, col] = C_coeffs[2][l]
            W[8+2*k+1, col] = C_coeffs[3][l]
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8) using Trivial
    # A2 indices: (0,2), (1,2) -> U indices 4, 5
    # B2 indices: (2,0)..(2,7) -> V indices 16..23
    # C indices: same as before
    
    for i in range(2):
        for j in range(8):
            # U: A2[i]
            U[4+i, col] = 1
            
            # V: B2[j]
            V[16+j, col] = 1
            
            # W: C[i,j]
            W[8*i+j, col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
