"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition using Strassen blocks for 2x2x2 parts
    # and schoolbook for the remainder.
    
    # Helper to create a 2x2x2 Strassen block
    # Returns U, V, W for n=2, m=2, p=2
    def get_strassen_block():
        r = 7
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 (0), A22 (3)
        # V: B11 (0), B22 (3)
        # W: C11 (0), C22 (3)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        # U: A21 (2), A22 (3)
        # V: B11 (0)
        # W: C21 (2), C22 (3)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A11(B21-B22)
        # U: A11 (0)
        # V: B21 (1), B22 (3)
        # W: C12 (1), C21 (2)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[2][2] = 1
        
        # M4 = A22(B12-B11)
        # U: A22 (3)
        # V: B12 (2), B11 (0)
        # W: C11 (0), C21 (2)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        # U: A11 (0), A12 (1)
        # V: B22 (3)
        # W: C11 (0), C12 (1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21 (2), A11 (0)
        # V: B11 (0), B12 (2)
        # W: C22 (3)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12 (1), A22 (3)
        # V: B21 (1), B22 (3)
        # W: C11 (0), C22 (3)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    U_s, V_s, W_s = get_strassen_block()
    
    # Initialize full matrices
    # We will build the decomposition block by block.
    # Total rank will be sum of ranks of blocks.
    
    # Block 1: i in [0,1], j in [0,1], k in [0,7]
    # This is 2x2x8. Composed of 4 Strassen blocks (k in [0,1], [2,3], [4,5], [6,7])
    # Block 2: i in [0,1], j in [2,3], k in [0,7]
    # This is 2x2x8. Composed of 4 Strassen blocks.
    # Block 3: i in [2,2], j in [0,3], k in [0,7]
    # This is 1x4x8. Schoolbook.
    
    rank1 = 4 * 7 # 28
    rank2 = 4 * 7 # 28
    rank3 = 1 * 4 * 8 # 32
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block at specific offsets
    def add_strassen_block(i_off, j_off, k_off, col_start):
        for c in range(7):
            # U indices
            for u_i in range(2):
                for u_j in range(2):
                    val = U_s[u_i * 2 + u_j][c]
                    if val != 0:
                        U[(i_off + u_i) * m + (j_off + u_j)][col_start + c] += val
            # V indices
            for v_j in range(2):
                for v_k in range(2):
                    val = V_s[v_j * 2 + v_k][c]
                    if val != 0:
                        V[(j_off + v_j) * p + (k_off + v_k)][col_start + c] += val
            # W indices
            for w_k in range(2):
                for w_i in range(2):
                    val = W_s[w_k * 2 + w_i][c]
                    if val != 0:
                        W[(k_off + w_k) * n + (i_off + w_i)][col_start + c] += val

    # Add Block 1 (4 Strassen blocks)
    for k_block in range(4):
        k_off = k_block * 2
        add_strassen_block(0, 0, k_off, col)
        col += 7
        
    # Add Block 2 (4 Strassen blocks)
    for k_block in range(4):
        k_off = k_block * 2
        add_strassen_block(0, 2, k_off, col)
        col += 7
        
    # Add Block 3 (Schoolbook 1x4x8)
    for i in range(1):
        for j in range(4):
            for k in range(8):
                U[(2 + i) * m + (0 + j)][col] = 1
                V[(0 + j) * p + (0 + k)][col] = 1
                W[(0 + k) * n + (2 + i)][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
