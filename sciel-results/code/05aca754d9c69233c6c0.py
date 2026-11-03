import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 27 (3x3 schoolbook) * 7 (2x2 Strassen) = 189
    rank = 189
    
    # Strassen decomposition for 2x2 matrices
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1)
    U_s = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    V_s = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    W_s = [
        [1, 0, 0, 1], # M1: C11+C22
        [0, 0, 1, -1],# M2: C21-C22
        [0, 1, 0, 1], # M3: C12+C22
        [1, 0, 1, 0], # M4: C11+C21
        [-1, 1, 0, 0],# M5: -C11+C12
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0]  # M7: C11
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # Process Strassen term s for block (I,J,K)
                    for u in range(2):
                        for v in range(2):
                            # A sub-index (u,v)
                            if U_s[s][u * 2 + v] != 0:
                                row_a = 2 * I + u
                                col_a = 2 * J + v
                                idx_a = row_a * m + col_a
                                U[idx_a][r] = U_s[s][u * 2 + v]
                            
                            # B sub-index (u,v) -> Note: Strassen V uses same sub-indices structure
                            # In Strassen, V is for B. B row is u, col is v.
                            if V_s[s][u * 2 + v] != 0:
                                row_b = 2 * J + u
                                col_b = 2 * K + v
                                idx_b = row_b * p + col_b
                                V[idx_b][r] = V_s[s][u * 2 + v]
                                
                            # C sub-index (u,v)
                            if W_s[s][u * 2 + v] != 0:
                                row_c = 2 * I + u
                                col_c = 2 * K + v
                                # Prompt convention: W[k*n+i] where k is col, i is row?
                                # Seed: W[k*n+i] for A_ij B_jk -> C_ik?
                                # Seed loop: i, j, k. W[k*n+i].
                                # So index is col_c * n + row_c.
                                idx_c = col_c * n + row_c
                                W[idx_c][r] = W_s[s][u * 2 + v]
                    r += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
