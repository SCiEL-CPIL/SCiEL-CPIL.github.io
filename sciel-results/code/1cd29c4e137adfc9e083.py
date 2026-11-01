import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients
    # u: list of dicts { (i,j): coeff }
    # v: list of dicts { (j,k): coeff }
    # w: list of dicts { (i,k): coeff }
    
    # Indices for x (2x2)
    # x00, x01, x10, x11
    # Indices for y (2x2)
    # y00, y01, y10, y11
    
    strassen_u = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    strassen_v = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    strassen_w = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for q in range(3):
        # Block q of B, columns 2q, 2q+1
        # A_L: rows 0,1 of A, cols 0,1 of A (j in 0,1)
        # A_R: rows 0,1 of A, cols 2,3 of A (j in 2,3)
        
        # Term for A_L * B_T,q
        for s in range(7):
            # U contributions
            for (i, j), coeff in strassen_u[s].items():
                # A_L uses j in 0,1
                U[i * m + j, col] += coeff
            
            # V contributions
            for (j, k), coeff in strassen_v[s].items():
                # B_T,q uses j in 0,1, k in 2q, 2q+1
                # y_jk corresponds to B_{j, 2q+k}
                V[j * p + (2 * q + k), col] += coeff
            
            # W contributions
            for (i, k), coeff in strassen_w[s].items():
                # C_q uses i in 0,1, k in 2q, 2q+1
                # w_ik corresponds to C_{i, 2q+k}
                W[(2 * q + k) * n + i, col] += coeff
            
            col += 1
        
        # Term for A_R * B_B,q
        for s in range(7):
            # U contributions
            for (i, j), coeff in strassen_u[s].items():
                # A_R uses j in 2,3
                # Map local j (0,1) to global j (2,3)
                U[i * m + (2 + j), col] += coeff
            
            # V contributions
            for (j, k), coeff in strassen_v[s].items():
                # B_B,q uses j in 2,3, k in 2q, 2q+1
                # Map local j (0,1) to global j (2,3)
                V[(2 + j) * p + (2 * q + k), col] += coeff
            
            # W contributions
            for (i, k), coeff in strassen_w[s].items():
                # C_q uses i in 0,1, k in 2q, 2q+1
                W[(2 * q + k) * n + i, col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
