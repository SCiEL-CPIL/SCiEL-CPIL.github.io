"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients
    # A coeffs for M0..M6
    A_coeffs = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, -1], 
        [0, 1, 0, -1]  
    ]
    
    # B coeffs (local) for M0..M6
    B_coeffs = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [0, 0, 1, -1], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    
    # C coeffs (local) for M0..M6
    # Columns are M0..M6, Rows are c00, c01, c10, c11
    C_coeffs_T = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    for l in range(6):
        for s in range(7):
            r = l * 7 + s
            
            # Fill U
            U[0:4, r] = A_coeffs[s]
            
            # Fill V
            V[2*l, r] = B_coeffs[s][0]
            V[2*l+1, r] = B_coeffs[s][1]
            V[12+2*l, r] = B_coeffs[s][2]
            V[13+2*l, r] = B_coeffs[s][3]
            
            # Fill W
            W[4*l, r] = C_coeffs_T[s][0]
            W[4*l+2, r] = C_coeffs_T[s][1]
            W[4*l+1, r] = C_coeffs_T[s][2]
            W[4*l+3, r] = C_coeffs_T[s][3]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
