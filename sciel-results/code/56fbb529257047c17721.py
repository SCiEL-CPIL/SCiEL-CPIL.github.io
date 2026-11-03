"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56: 2 blocks * 4 k-pairs * 7 Strassen terms
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2x2
    # U: (x00, x01, x10, x11)
    # V: (y00, y01, y10, y11)
    # W: (z00, z01, z10, z11)
    strassen_U = [
        [1, 0, 0, 1],  # M1: x00 + x11
        [0, 0, 1, 1],  # M2: x10 + x11
        [1, 0, 0, 0],  # M3: x00
        [0, 0, 0, 1],  # M4: x11
        [1, 1, 0, 0],  # M5: x00 + x01
        [-1, 0, 1, 0], # M6: x10 - x00
        [0, 1, 0, -1]  # M7: x01 - x11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1: y00 + y11
        [1, 0, 0, 0],  # M2: y00
        [0, 1, 0, -1], # M3: y01 - y11
        [-1, 0, 1, 0], # M4: y10 - y00
        [0, 0, 0, 1],  # M5: y11
        [1, 1, 0, 0],  # M6: y00 + y01
        [0, 0, 1, 1]   # M7: y10 + y11
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # M1: z00 + z11
        [0, 0, 1, -1], # M2: z10 - z11
        [0, 1, 0, 1],  # M3: z01 + z11
        [1, 0, 1, 1],  # M4: z00 + z10 + z11
        [-1, 1, 0, 0], # M5: -z00 + z01
        [0, 0, 0, 1],  # M6: z11
        [1, 0, 0, 0]   # M7: z00
    ]
    
    col = 0
    # Two blocks: j in {0,1} and j in {2,3}
    for block in range(2):
        # 4 k-pairs: k in {0,1}, {2,3}, {4,5}, {6,7}
        for q in range(4):
            # 7 Strassen terms
            for s in range(7):
                # U indices
                # Local x00 -> i=0, j=block*2+0
                # Local x01 -> i=0, j=block*2+1
                # Local x10 -> i=1, j=block*2+0
                # Local x11 -> i=1, j=block*2+1
                
                u_vals = strassen_U[s]
                u_rows = [
                    0 * m + (block * 2 + 0), # x00
                    0 * m + (block * 2 + 1), # x01
                    1 * m + (block * 2 + 0), # x10
                    1 * m + (block * 2 + 1)  # x11
                ]
                
                for idx, val in enumerate(u_vals):
                    if val != 0:
                        U[u_rows[idx]][col] += val
                        
                # V indices
                # Local y00 -> j=block*2+0, k=2*q+0
                # Local y01 -> j=block*2+0, k=2*q+1
                # Local y10 -> j=block*2+1, k=2*q+0
                # Local y11 -> j=block*2+1, k=2*q+1
                
                v_vals = strassen_V[s]
                v_rows = [
                    (block * 2 + 0) * p + (2 * q + 0), # y00
                    (block * 2 + 0) * p + (2 * q + 1), # y01
                    (block * 2 + 1) * p + (2 * q + 0), # y10
                    (block * 2 + 1) * p + (2 * q + 1)  # y11
                ]
                
                for idx, val in enumerate(v_vals):
                    if val != 0:
                        V[v_rows[idx]][col] += val
                        
                # W indices
                # Local z00 -> k=2*q+0, i=0
                # Local z01 -> k=2*q+1, i=0
                # Local z10 -> k=2*q+0, i=1
                # Local z11 -> k=2*q+1, i=1
                # Convention: W[k*n + i]
                
                w_vals = strassen_W[s]
                w_rows = [
                    (2 * q + 0) * n + 0, # z00
                    (2 * q + 1) * n + 0, # z01
                    (2 * q + 0) * n + 1, # z10
                    (2 * q + 1) * n + 1  # z11
                ]
                
                for idx, val in enumerate(w_vals):
                    if val != 0:
                        W[w_rows[idx]][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
