"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen factors
    # U2: rows correspond to A00, A01, A10, A11
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ])
    
    # V2: rows correspond to B00, B01, B10, B11
    V2 = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    
    # W2: rows correspond to C00, C01, C10, C11
    W2 = np.array([
        [1, 0, -1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, -1, 0, 1, 0]
    ])
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Precompute non-zero entries for efficiency
    # For each row in 0..3, store list of (col, val)
    U2_nz = [[(c, U2[r, c]) for c in range(7) if U2[r, c] != 0] for r in range(4)]
    V2_nz = [[(c, V2[r, c]) for c in range(7) if V2[r, c] != 0] for r in range(4)]
    W2_nz = [[(c, W2[r, c]) for c in range(7) if W2[r, c] != 0] for r in range(4)]
    
    # Construct U
    for idx in range(n * m):
        r = idx // m
        c = idx % m
        I = r // 2
        J = c // 2
        u = r % 2
        v = c % 2
        
        row_b = I * 2 + J
        row_e = u * 2 + v
        
        for c1, val1 in U2_nz[row_b]:
            for c2, val2 in U2_nz[row_e]:
                col = c1 * 7 + c2
                U[idx, col] = val1 * val2
                
    # Construct V
    for idx in range(m * p):
        r = idx // p
        c = idx % p
        I = r // 2
        J = c // 2
        u = r % 2
        v = c % 2
        
        row_b = I * 2 + J
        row_e = u * 2 + v
        
        for c1, val1 in V2_nz[row_b]:
            for c2, val2 in V2_nz[row_e]:
                col = c1 * 7 + c2
                V[idx, col] = val1 * val2
                
    # Construct W
    for idx in range(p * n):
        r = idx // n
        c = idx % n
        I = r // 2
        J = c // 2
        u = r % 2
        v = c % 2
        
        row_b = I * 2 + J
        row_e = u * 2 + v
        
        for c1, val1 in W2_nz[row_b]:
            for c2, val2 in W2_nz[row_e]:
                col = c1 * 7 + c2
                W[idx, col] = val1 * val2
                
    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
