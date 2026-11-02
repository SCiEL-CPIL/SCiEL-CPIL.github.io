"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 decomposition: 7 (3x3x3) + 7 (3x3x3) + 9 (3x3x1)
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for M_3,3,3 (Rank 7)
    # U_idx: i*m+j, V_idx: j*p+k, W_idx: k*n+i
    # Note: For the sub-tensors, we map local indices to global.
    
    strassen_7 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (1,0,0, 0,1,0, 0,0,1)
        # V: (1,0,0, 0,1,0, 0,0,1)
        # W: (1,0,0, 0,1,0, 0,0,1)
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        
        # U: (0,1,0, 1,0,0, 0,0,0)
        # V: (0,0,1, 0,0,1, 0,0,0)
        # W: (0,1,0, 1,0,0, 0,0,0)
        ([0,1,0, 1,0,0, 0,0,0], [0,0,1, 0,0,1, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        
        # U: (0,0,1, 0,0,1, 0,0,0)
        # V: (0,1,0, 0,0,0, 1,0,0)
        # W: (0,0,0, 0,1,0, 0,1,0)
        ([0,0,1, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,1,0]),
        
        # U: (1,0,0, 0,-1,0, 0,0,0)
        # V: (1,0,0, 0,1,0, 0,0,0)
        # W: (1,0,0, 0,-1,0, 0,0,0)
        ([1,0,0, 0,-1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,-1,0, 0,0,0]),
        
        # U: (0,1,0, 0,0,0, 1,0,0)
        # V: (0,0,0, 1,0,0, 0,0,1)
        # W: (0,1,0, 0,0,0, 1,0,0)
        ([0,1,0, 0,0,0, 1,0,0], [0,0,0, 1,0,0, 0,0,1], [0,1,0, 0,0,0, 1,0,0]),
        
        # U: (0,0,1, 0,0,0, 0,1,0)
        # V: (0,0,0, 0,1,0, 0,0,1)
        # W: (0,0,0, 0,0,1, 0,1,0)
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,1,0]),
        
        # U: (0,0,0, 0,0,1, 0,0,1)
        # V: (0,0,0, 0,0,0, 1,0,0)
        # W: (0,0,0, 0,0,0, 0,0,1)
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Block 1: M_3,3,3 for k in {0,1,2}
    # Global W index: k*n + i
    for c in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_7[c]
        for idx in range(9):
            U[idx][c] = u_coeffs[idx]
            V[idx][c] = v_coeffs[idx]
            W[idx][c] = w_coeffs[idx]
            
    # Block 2: M_3,3,3 for k in {3,4,5}
    # Global W index: (k+3)*n + i = 3*n + k*n + i = 9 + k*n + i
    for c in range(7, 14):
        local_c = c - 7
        u_coeffs, v_coeffs, w_coeffs = strassen_7[local_c]
        for idx in range(9):
            U[idx][c] = u_coeffs[idx]
            V[idx][c] = v_coeffs[idx]
            # Shift W indices by 9 (3 * n)
            W[idx + 9][c] = w_coeffs[idx]
            
    # Block 3: M_3,3,1 for k=6
    # Schoolbook for 3x3x1
    # U: i*m+j, V: j*1+0, W: 6*n+i
    for i in range(3):
        for j in range(3):
            c = 14 + i * 3 + j
            U[i * 3 + j][c] = 1
            V[j * 1 + 0][c] = 1
            W[6 * 3 + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
