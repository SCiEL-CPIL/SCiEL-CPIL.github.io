"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 decomposition matrices
    # Rows correspond to matrix entries in row-major order (00, 01, 10, 11)
    # Columns correspond to the 7 rank-1 terms
    
    # U (A-side)
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V (B-side)
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 1],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W (C-side)
    # Original row-major: C00, C01, C10, C11
    # Task requires index k*n+i. For 2x2, k*2+i.
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    # So we need to permute rows: 0, 2, 1, 3
    S_W_orig = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    S_W = [
        S_W_orig[0], # C00
        S_W_orig[2], # C10
        S_W_orig[1], # C01
        S_W_orig[3]  # C11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the tensor product decomposition
    for r in range(7):
        for s in range(7):
            c = 7 * r + s
            # Iterate over block indices and element indices
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            # A indices
                            R = 2 * I + i
                            C = 2 * J + j
                            u_block = S_U[I * 2 + J][r]
                            u_elem = S_U[i * 2 + j][s]
                            U[R * m + C][c] = u_block * u_elem
                            
                            # B indices
                            # B is m x p. Row j_glob, Col k_glob
                            # j_glob = 2*J + j
                            # k_glob = 2*K + k
                            # We need to sum over K, k as well?
                            # No, the tensor product structure implies:
                            # V term depends on J, K for block and j, k for elem.
                            # We must iterate K, k separately for V and W.
                            pass
                            
            # Correct approach: Iterate all indices for U, V, W independently?
            # No, the rank-1 term is a product of vectors.
            # U vector components are fixed for (r, s).
            # V vector components are fixed for (r, s).
            # W vector components are fixed for (r, s).
            
            # Let's fill U, V, W for this column c.
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            R = 2 * I + i
                            C = 2 * J + j
                            U[R * m + C][c] = S_U[I * 2 + J][r] * S_U[i * 2 + j][s]
                            
            # Fill V
            for J in range(2):
                for K in range(2):
                    for j in range(2):
                        for k in range(2):
                            R = 2 * J + j
                            C = 2 * K + k
                            V[R * p + C][c] = S_V[J * 2 + K][r] * S_V[j * 2 + k][s]
                            
            # Fill W
            for K in range(2):
                for I in range(2):
                    for k in range(2):
                        for i in range(2):
                            R = 2 * K + k
                            C = 2 * I + i
                            # W index is k_glob * n + i_glob
                            # k_glob = 2*K + k
                            # i_glob = 2*I + i
                            W[R * n + C][c] = S_W[K * 2 + I][r] * S_W[k * 2 + i][s]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
