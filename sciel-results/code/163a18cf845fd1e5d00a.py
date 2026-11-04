"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 30 decomposition using mixed Schoolbook (2x2x4) and Strassen (2x2x2)
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    def add_schoolbook_block_2x2x4(i_start, j_start, k_start, offset):
        # Adds a 2x2x4 Schoolbook block
        # A: 2x2, B: 2x4, C: 2x4
        # 8 multiplications
        for k_offset in range(4):
            k = k_start + k_offset
            
            # C[0, k] += A[0, 0] * B[0, k]
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + k][col] += 1
            W[k*n + (i_start+0)][col] += 1
            offset += 1
            
            # C[0, k] += A[0, 1] * B[1, k]
            col = offset
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + k][col] += 1
            W[k*n + (i_start+0)][col] += 1
            offset += 1
            
            # C[1, k] += A[1, 0] * B[0, k]
            col = offset
            U[(i_start+1)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + k][col] += 1
            W[k*n + (i_start+1)][col] += 1
            offset += 1
            
            # C[1, k] += A[1, 1] * B[1, k]
            col = offset
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + k][col] += 1
            W[k*n + (i_start+1)][col] += 1
            offset += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    # B[0:2, 0:6] split into B[0:2, 0:4] (2x4) and B[0:2, 4:6] (2x2)
    add_schoolbook_block_2x2x4(0, 0, 0, 0)   # Rank 8
    add_strassen_block(0, 0, 4, 8)           # Rank 7
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:6]
    # B[2:4, 0:6] split into B[2:4, 0:4] (2x4) and B[2:4, 4:6] (2x2)
    add_schoolbook_block_2x2x4(0, 2, 0, 15)  # Rank 8
    add_strassen_block(0, 2, 4, 23)          # Rank 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
