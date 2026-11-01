import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p // 2) = 7 * 7 = 49
    rank = 7 * (p // 2)
    
    # Initialize matrices
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # t=0: a+d, t=1: c+d, t=2: a, t=3: d, t=4: a+b, t=5: c-a, t=6: b-d
    # Indices: (i,j)
    A_coeffs = [
        [(0,0,1), (1,1,1)],       # t=0
        [(1,0,1), (1,1,1)],       # t=1
        [(0,0,1)],                # t=2
        [(1,1,1)],                # t=3
        [(0,0,1), (0,1,1)],       # t=4
        [(1,0,1), (0,0,-1)],      # t=5
        [(0,1,1), (1,1,-1)]       # t=6
    ]
    
    # Strassen coefficients for B (2x2 block)
    # t=0: x+w, t=1: x, t=2: y-w, t=3: z-x, t=4: w, t=5: x+z, t=6: y+w
    # Block cols: k0, k1
    # Rows: j=0, j=1
    # x=B[0,k0], y=B[0,k1], z=B[1,k0], w=B[1,k1]
    B_coeffs = [
        [(0,0,1), (1,1,1)],       # t=0: x + w
        [(0,0,1)],                # t=1: x
        [(0,1,1), (1,1,-1)],      # t=2: y - w
        [(1,0,1), (0,0,-1)],      # t=3: z - x
        [(1,1,1)],                # t=4: w
        [(0,0,1), (1,0,1)],       # t=5: x + z
        [(0,1,1), (1,1,1)]        # t=6: y + w
    ]
    
    # Reconstruction coefficients for C
    # C[0,0] = P0 - P3 + P5 - P6
    # C[0,1] = P2 + P6
    # C[1,0] = P1 + P3
    # C[1,1] = P0 - P1 + P2 + P4
    # Indices (i, local_k) -> list of (t, coeff)
    C_coeffs = [
        { (0,0): [(0,1), (3,-1), (5,1), (6,-1)],
          (0,1): [(2,1), (6,1)],
          (1,0): [(1,1), (3,1)],
          (1,1): [(0,1), (1,-1), (2,1), (4,1)] }
    ]
    
    for l in range(p // 2):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        for t in range(7):
            r = l * 7 + t
            
            # Fill U
            for (i, j, val) in A_coeffs[t]:
                U[i * m + j][r] = val
                
            # Fill V
            for (j, local_k, val) in B_coeffs[t]:
                # local_k 0 -> k0, 1 -> k1
                k = k0 if local_k == 0 else k1
                V[j * p + k][r] = val
                
            # Fill W
            # C_coeffs is a list of one dict for 2x2 block
            for (i, local_k, terms) in [(0,0,C_coeffs[0][(0,0)]), (0,1,C_coeffs[0][(0,1)]), (1,0,C_coeffs[0][(1,0)]), (1,1,C_coeffs[0][(1,1)])]:
                k = k0 if local_k == 0 else k1
                for (t_val, coeff) in terms:
                    if t_val == t:
                        W[k * n + i][r] = coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
