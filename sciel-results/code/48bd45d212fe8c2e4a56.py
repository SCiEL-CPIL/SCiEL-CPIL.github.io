"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: a00, a01, a10, a11
    # v: b00, b01, b10, b11
    # w: c00, c01, c10, c11
    strassen = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)), # P1
        ((0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 1, -1)), # P2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)), # P3
        ((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 1, 0)), # P4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)), # P5
        ((-1, 0, 1, 0), (1, 0, 1, 0), (0, 0, 0, 1)), # P6
        ((0, 1, 0, -1), (0, 1, 0, 1), (1, 0, 0, 1))  # P7
    ]
    
    for b in range(6):
        if time.time() > deadline:
            break
            
        col_base = b * 7
        
        # Determine A indices
        if b < 3:
            a_idx = [0, 1, 4, 5]
        else:
            a_idx = [2, 3, 6, 7]
            
        # Determine B indices
        row_start = 0 if b < 3 else 2
        col_start = (b % 3) * 2
        b_idx = [
            row_start * 6 + col_start,
            row_start * 6 + col_start + 1,
            (row_start + 1) * 6 + col_start,
            (row_start + 1) * 6 + col_start + 1
        ]
        
        # Determine C indices (W indices)
        k_start = (b % 3) * 2
        w_idx = [
            k_start * 2 + 0,
            (k_start + 1) * 2 + 0,
            k_start * 2 + 1,
            (k_start + 1) * 2 + 1
        ]
        
        for s in range(7):
            u, v, w = strassen[s]
            col = col_base + s
            
            for i in range(4):
                U[a_idx[i]][col] = u[i]
                V[b_idx[i]][col] = v[i]
                W[w_idx[i]][col] = w[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
