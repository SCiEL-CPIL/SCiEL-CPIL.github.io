import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # U shape: (n*m, R) = (4, R)
    # V shape: (m*p, R) = (26, R)
    # W shape: (p*n, R) = (26, R)
    
    # We use 6 Strassen blocks (rank 7 each) + 1 trivial block (rank 4)
    # Total rank = 6 * 7 + 4 = 46
    
    R = 46
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # u vectors (coeffs for A)
    u_strassen = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [1, 0, -1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # v vectors (coeffs for B block)
    v_strassen = [
        [1, 1, 1, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # w vectors (coeffs for C block)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_strassen = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1],# M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0],# M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    col = 0
    # Process 6 full 2x2 blocks of B
    for s in range(6):
        # B columns: 2s, 2s+1
        # C columns: 2s, 2s+1
        
        # Map local B indices (0,1,2,3) to global V indices
        # V index = row * p + col
        # Local 0 (0,0) -> row 0, col 2s -> 0*13 + 2s = 2s
        # Local 1 (0,1) -> row 0, col 2s+1 -> 2s+1
        # Local 2 (1,0) -> row 1, col 2s -> 1*13 + 2s = 13+2s
        # Local 3 (1,1) -> row 1, col 2s+1 -> 13+2s+1
        v_global_idx = [2*s, 2*s+1, 13+2*s, 13+2*s+1]
        
        # Map local C indices (0,1,2,3) to global W indices
        # W index = col * n + row
        # Local 0 (C00) -> col 2s, row 0 -> 2s*2 + 0 = 4s
        # Local 1 (C01) -> col 2s+1, row 0 -> (2s+1)*2 + 0 = 4s+2
        # Local 2 (C10) -> col 2s, row 1 -> 2s*2 + 1 = 4s+1
        # Local 3 (C11) -> col 2s+1, row 1 -> (2s+1)*2 + 1 = 4s+3
        w_global_idx = [4*s, 4*s+2, 4*s+1, 4*s+3]
        
        for k in range(7):
            # Fill U
            for i in range(4):
                U[i][col] = u_strassen[k][i]
            
            # Fill V
            for l in range(4):
                V[v_global_idx[l]][col] = v_strassen[k][l]
            
            # Fill W
            for q in range(4):
                W[w_global_idx[q]][col] = w_strassen[k][q]
                
            col += 1
            
    # Process last block: B col 12, C col 12
    # Trivial decomposition for 2x2x1
    # C0,12 = A0,0 B0,12 + A0,1 B1,12
    # C1,12 = A1,0 B0,12 + A1,1 B1,12
    
    # Term 1: A0,0 B0,12 -> C0,12
    U[0][col] = 1
    V[0*13 + 12][col] = 1
    W[12*2 + 0][col] = 1
    col += 1
    
    # Term 2: A0,1 B1,12 -> C0,12
    U[1][col] = 1
    V[1*13 + 12][col] = 1
    W[12*2 + 0][col] = 1
    col += 1
    
    # Term 3: A1,0 B0,12 -> C1,12
    U[2][col] = 1
    V[0*13 + 12][col] = 1
    W[12*2 + 1][col] = 1
    col += 1
    
    # Term 4: A1,1 B1,12 -> C1,12
    U[3][col] = 1
    V[1*13 + 12][col] = 1
    W[12*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
