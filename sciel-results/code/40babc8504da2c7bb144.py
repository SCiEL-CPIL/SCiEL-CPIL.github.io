"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 steps for 2x2x2x2
    # Each step: (A_coeffs, B_coeffs, C_coeffs) as dict {idx: val}
    # A idx: 0=A00, 1=A01, 2=A10, 3=A11
    # B idx: 0=B00, 1=B01, 2=B10, 3=B11
    # C idx: 0=C00, 1=C01, 2=C10, 3=C11
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1},      {2:1, 3:-1}),
        ({0:1},      {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1},      {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1},      {0:-1, 1:1}),
        ({2:1, 0:-1},{0:1, 1:1},  {3:1}),
        ({1:1, 3:-1},{2:1, 3:1},  {0:1}),
    ]
    
    col = 0
    # A is 2x4: split into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3)
    # B is 4x8: split rows into top-2 and bottom-2, cols into 4 groups of 2
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13]]
    # C is 2x8: split into 4 groups of 2
    # C_j = A_L * B0j + A_R * B1j
    
    for half in range(2):  # 0: A_L, 1: A_R
        for cblock in range(4):  # column block of C
            for s in range(7):
                a_coeff, b_coeff, c_coeff = strassen[s]
                
                # A: 2x2 block at columns [2*half, 2*half+2)
                # Global A index: i*m + j where i in {0,1}, j in {2*half, 2*half+1}
                for a_idx, val in a_coeff.items():
                    i = a_idx // 2
                    j_loc = a_idx % 2
                    j_glob = 2 * half + j_loc
                    U[i * m + j_glob][col] += val
                
                # B: 2x2 block at rows [2*half, 2*half+2), cols [2*cblock, 2*cblock+2)
                for b_idx, val in b_coeff.items():
                    j_loc = b_idx // 2
                    k_loc = b_idx % 2
                    j_glob = 2 * half + j_loc
                    k_glob = 2 * cblock + k_loc
                    V[j_glob * p + k_glob][col] += val
                
                # C: 2x2 block at cols [2*cblock, 2*cblock+2)
                for c_idx, val in c_coeff.items():
                    i = c_idx // 2
                    k_loc = c_idx % 2
                    k_glob = 2 * cblock + k_loc
                    W[k_glob * n + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
