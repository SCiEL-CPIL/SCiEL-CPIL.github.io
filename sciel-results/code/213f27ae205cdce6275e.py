"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition:
    # 28 terms from 4 Strassen 2x2x2 blocks for the 4x4 part of B
    # 16 terms from schoolbook 4x4x1 for the 4x1 part of B
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for A (4x4) * B_left (4x4) ---
    # B_left consists of columns 0-3 of B.
    # We split A and B_left into 2x2 blocks.
    
    # Strassen multipliers definitions for 2x2 block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coeffs for A block (local indices a,b)
    # V coeffs for B block (local indices b,d)
    # W coeffs for C block (local indices a,d)
    
    strassen_U = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    strassen_V = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    strassen_W = [
        { (0,0): 1, (1,1): 1 }, # M1 -> C00, C11
        { (1,0): 1 },           # M2 -> C10
        { (0,1): 1 },           # M3 -> C01
        { (0,0): 1, (1,0): 1 }, # M4 -> C00, C10
        { (0,0): -1 },          # M5 -> C00 (-1)
        { (1,1): 1 },           # M6 -> C11
        { (0,0): 1 }            # M7 -> C00
    ]
    # Wait, let's re-verify W mapping for M5, M6, M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00+1, C11+1
    # M2: C10+1
    # M3: C01+1
    # M4: C00+1, C10+1
    # M5: C00-1, C01+1
    # M6: C11+1
    # M7: C00+1
    
    strassen_W_corrected = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1 },           # M2
        { (0,1): 1 },           # M3
        { (0,0): 1, (1,0): 1 }, # M4
        { (0,0): -1, (0,1): 1 },# M5
        { (1,1): 1 },           # M6
        { (0,0): 1 }            # M7
    ]
    
    col = 0
    for r in range(2): # Block row
        for c in range(2): # Block col
            for s in range(7): # Strassen multiplier
                # U updates
                for (a, b), val in strassen_U[s].items():
                    i = 2*r + a
                    j = 2*c + b
                    U[i*m + j][col] += val
                
                # V updates
                for (b, d), val in strassen_V[s].items():
                    j = 2*c + b
                    k = 2*c + d # B_left col index
                    V[j*p + k][col] += val
                
                # W updates
                for (a, d), val in strassen_W_corrected[s].items():
                    i = 2*r + a
                    k = 2*c + d
                    W[k*n + i][col] += val
                
                col += 1
    
    # --- Part 2: Schoolbook for A (4x4) * B_right (4x1) ---
    # B_right is column 4 of B.
    # C_right is column 4 of C.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    
    for i in range(4):
        for j in range(4):
            # U
            U[i*m + j][col] = 1
            # V
            V[j*p + 4][col] = 1
            # W
            W[4*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
