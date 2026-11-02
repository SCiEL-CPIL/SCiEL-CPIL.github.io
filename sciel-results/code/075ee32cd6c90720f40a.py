"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # TL (1x3x2): 6
    # TR (1x2x4): 8
    # BL (2x3x2): 4 (2x1x2) + 7 (2x2x2) = 11
    # BR (2x2x4): 14 (two 2x2x2)
    # Total: 6 + 8 + 11 + 14 = 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block TL: i in [0,1), j in [0,3), k in [0,2)
    for j in range(3):
        for k in range(2):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Block TR: i in [0,1), j in [3,5), k in [2,6)
    for j_loc in range(2):
        j = 3 + j_loc
        for k_loc in range(4):
            k = 2 + k_loc
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Block BL: i in [1,3), j in [0,3), k in [0,2)
    # Part 1: j=0 (2x1x2)
    for i_loc in range(2):
        i = 1 + i_loc
        for k in range(2):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in [1,3) (2x2x2) - Strassen
    strassen_terms = [
        # (U_mat, V_mat, W_mat)
        ([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]),
        ([0,0, 1,1], [1,0, 0,0], [1,0, 0,0]),
        ([1,0, 0,0], [0,-1, 0,0], [0,0, 0,1]),
        ([0,0, 0,1], [0,1, 0,-1], [0,0, 0,1]),
        ([1,1, 0,0], [0,0, 1,0], [0,0, 0,1]),
        ([-1,0, 1,0], [1,1, 0,0], [0,0, 0,1]),
        ([0,1, 0,-1], [0,1, 1,1], [0,0, 0,1]),
    ]
    
    for u_mat, v_mat, w_mat in strassen_terms:
        for i_loc in range(2):
            i = 1 + i_loc
            for j_loc in range(2):
                j = 1 + j_loc
                val_u = u_mat[i_loc * 2 + j_loc]
                if val_u != 0:
                    U[i * m + j][col] += val_u
        for j_loc in range(2):
            j = 1 + j_loc
            for k_loc in range(2):
                k = k_loc
                val_v = v_mat[j_loc * 2 + k_loc]
                if val_v != 0:
                    V[j * p + k][col] += val_v
        for k_loc in range(2):
            k = k_loc
            for i_loc in range(2):
                i = 1 + i_loc
                val_w = w_mat[k_loc * 2 + i_loc]
                if val_w != 0:
                    W[k * n + i][col] += val_w
        col += 1
        
    # Block BR: i in [1,3), j in [3,5), k in [2,6)
    # Size 2x2x4. Split k into [2,4) and [4,6).
    # Two 2x2x2 blocks.
    for k_block in range(2):
        k_offset = 2 + k_block * 2
        for i_loc in range(2):
            i = 1 + i_loc
            for j_loc in range(2):
                j = 3 + j_loc
                # This is just schoolbook for 2x2x2?
                # No, we can use Strassen here too.
                # But 2x2x4 can be done with 14 ranks (7+7).
                # We will use Strassen for each 2x2x2 part.
                pass
                
    # We already used Strassen for BL.
    # For BR, we have two 2x2x2 blocks.
    # Block 1: k in [2,4)
    # Block 2: k in [4,6)
    
    for k_block in range(2):
        k_offset = 2 + k_block * 2
        for u_mat, v_mat, w_mat in strassen_terms:
            for i_loc in range(2):
                i = 1 + i_loc
                for j_loc in range(2):
                    j = 3 + j_loc
                    val_u = u_mat[i_loc * 2 + j_loc]
                    if val_u != 0:
                        U[i * m + j][col] += val_u
            for j_loc in range(2):
                j = 3 + j_loc
                for k_loc in range(2):
                    k = k_offset + k_loc
                    val_v = v_mat[j_loc * 2 + k_loc]
                    if val_v != 0:
                        V[j * p + k][col] += val_v
            for k_loc in range(2):
                k = k_offset + k_loc
                for i_loc in range(2):
                    i = 1 + i_loc
                    val_w = w_mat[k_loc * 2 + i_loc]
                    if val_w != 0:
                        W[k * n + i][col] += val_w
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
