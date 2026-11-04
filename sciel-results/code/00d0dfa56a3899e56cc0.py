"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Block 1: M_{1,5,6} (Row 0 of A) ---
    # Rank 30
    col = 0
    for j in range(5):
        for k in range(6):
            U[0 * 5 + j][col] = 1
            V[j * 6 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # --- Block 2: M_{2,5,6} (Rows 1,2 of A) ---
    # Decomposed into 6 Strassen blocks (M_{2,2,2}) and 3 Schoolbook blocks (M_{2,1,2})
    
    # Strassen Matrices (Local 4x7)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook M_{2,1,2} Matrices (Local 2x4, 2x4, 4x4)
    U_sb = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    V_sb = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sb = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    # Segments for j (5) and k (6)
    j_segs = [[0, 1], [2, 3], [4]]
    k_segs = [[0, 1], [2, 3], [4, 5]]
    
    for js_idx, j_seg in enumerate(j_segs):
        for ks_idx, k_seg in enumerate(k_segs):
            if len(j_seg) == 2 and len(k_seg) == 2:
                # Strassen Block
                j0, j1 = j_seg
                k0, k1 = k_seg
                # Local to Global Mapping
                # A: (1, j0)->0, (1, j1)->1, (2, j0)->2, (2, j1)->3
                # B: (j0, k0)->0, (j0, k1)->1, (j1, k0)->2, (j1, k1)->3
                # C: (k0, 1)->0, (k1, 1)->1, (k0, 2)->2, (k1, 2)->3
                
                a_map = [1 * 5 + j0, 1 * 5 + j1, 2 * 5 + j0, 2 * 5 + j1]
                b_map = [j0 * 6 + k0, j0 * 6 + k1, j1 * 6 + k0, j1 * 6 + k1]
                c_map = [k0 * 3 + 1, k1 * 3 + 1, k0 * 3 + 2, k1 * 3 + 2]
                
                for r in range(7):
                    for la in range(4):
                        U[a_map[la]][col + r] += U_strassen[la][r]
                    for lb in range(4):
                        V[b_map[lb]][col + r] += V_strassen[lb][r]
                    for lc in range(4):
                        W[c_map[lc]][col + r] += W_strassen[lc][r]
                col += 7
            else:
                # Schoolbook Block M_{2,1,2}
                j0 = j_seg[0]
                k0, k1 = k_seg
                # A: (1, j0)->0, (2, j0)->1
                # B: (j0, k0)->0, (j0, k1)->1
                # C: (k0, 1)->0, (k1, 1)->1, (k0, 2)->2, (k1, 2)->3
                
                a_map = [1 * 5 + j0, 2 * 5 + j0]
                b_map = [j0 * 6 + k0, j0 * 6 + k1]
                c_map = [k0 * 3 + 1, k1 * 3 + 1, k0 * 3 + 2, k1 * 3 + 2]
                
                for r in range(4):
                    for la in range(2):
                        U[a_map[la]][col + r] += U_sb[la][r]
                    for lb in range(2):
                        V[b_map[lb]][col + r] += V_sb[lb][r]
                    for lc in range(4):
                        W[c_map[lc]][col + r] += W_sb[lc][r]
                col += 4
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
