"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1: (x11+x22)(y11+y22)
    # M2: (x11+x12)y22
    # M3: x11(y11-y21)
    # M4: x22(y22-y12)
    # M5: (x11+x21)y11
    # M6: (x12-x22)(y21+y22)
    # M7: (x21-x11)(y11+y12)
    
    # U coeffs: A[i,j]
    # V coeffs: B[j,k] (local k)
    # W coeffs: C[i,k] (local k)
    
    # Indices for A (2x2): 00, 01, 10, 11
    # Indices for B_s (2x2): 00, 01, 10, 11
    # Indices for C_s (2x2): 00, 01, 10, 11
    
    # We map local indices to global indices
    # A is global, so A[i,j] is A[i,j]
    # B_s is a block of B. B[j, 2s + k_loc]
    # C_s is a block of C. C[i, 2s + k_loc]
    
    # Strassen definitions
    # m: 1..7
    # u_coeffs: list of (i, j, coeff)
    # v_coeffs: list of (j, k_loc, coeff)
    # w_coeffs: list of (i, k_loc, coeff)
    
    strassen_defs = [
        # M1
        {
            'u': [(0,0,1), (1,1,1)],
            'v': [(0,0,1), (1,1,1)],
            'w': [(0,0,1), (1,1,1)] # z11 has M1, z22 has M1
        },
        # M2
        {
            'u': [(0,0,1), (0,1,1)],
            'v': [(1,1,1)],
            'w': [(0,1,1), (1,1,-1)] # z12 has M2, z22 has -M2
        },
        # M3
        {
            'u': [(0,0,1)],
            'v': [(0,0,1), (1,0,-1)],
            'w': [(0,1,1), (1,1,1)] # z12 has M3, z22 has M3
        },
        # M4
        {
            'u': [(1,1,1)],
            'v': [(1,1,1), (0,1,-1)],
            'w': [(0,0,1), (1,1,1)] # z11 has M4, z22 has M4
        },
        # M5
        {
            'u': [(0,0,1), (1,0,1)],
            'v': [(0,0,1)],
            'w': [(0,0,-1), (1,0,1)] # z11 has -M5, z21 has M5
        },
        # M6
        {
            'u': [(0,1,1), (1,1,-1)],
            'v': [(1,0,1), (1,1,1)],
            'w': [(1,0,1)] # z21 has M6
        },
        # M7
        {
            'u': [(1,0,1), (0,0,-1)],
            'v': [(0,0,1), (0,1,1)],
            'w': [(0,0,1)] # z11 has M7
        }
    ]
    
    for s in range(3):
        for m_idx in range(7):
            r = s * 7 + m_idx
            defn = strassen_defs[m_idx]
            
            # Fill U
            for i, j, coeff in defn['u']:
                U[i * m + j][r] += coeff
            
            # Fill V
            for j, k_loc, coeff in defn['v']:
                k_global = 2 * s + k_loc
                V[j * p + k_global][r] += coeff
            
            # Fill W
            for i, k_loc, coeff in defn['w']:
                k_global = 2 * s + k_loc
                # W index is k*n + i
                W[k_global * n + i][r] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
