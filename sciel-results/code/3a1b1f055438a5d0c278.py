"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 2x1x6 (j=0)
    # i in {0,1}, j=0, k in {0..5}
    col = 0
    for i in range(2):
        for k in range(6):
            U[5*i + 0][col] = 1
            V[0*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    # col is now 12

    # Helper to add a Strassen block for 2x2x2
    # i_offset: 0 or 1 (not used directly, i is 0..1)
    # j_offset: 1 or 3 (for j in {1,2} or {3,4})
    # k_offset: 0, 2, 4 (for k pairs)
    # col_start: starting column index
    def add_strassen_block(j_offset, k_offset, col_start):
        # j' in {0,1} maps to j = j_offset + j'
        # k' in {0,1} maps to k = k_offset + k'
        # i in {0,1}
        
        # Strassen terms
        # r=1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=2
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] -= 1
        
        # r=3
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=4
        col_start += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] -= 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        
        # r=5
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        
        # r=6
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 0][col_start] -= 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=7
        col_start += 1
        U[5*0 + j_offset + 1][col_start] += 1
        U[5*1 + j_offset + 1][col_start] -= 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        
        return col_start + 1

    # Block 2: j in {1,2}
    # k pairs: 0-1, 2-3, 4-5
    col = add_strassen_block(1, 0, 12)
    col = add_strassen_block(1, 2, col)
    col = add_strassen_block(1, 4, col)
    # col is now 12 + 21 = 33

    # Block 3: j in {3,4}
    col = add_strassen_block(3, 0, col)
    col = add_strassen_block(3, 2, col)
    col = add_strassen_block(3, 4, col)
    # col is now 33 + 21 = 54

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
