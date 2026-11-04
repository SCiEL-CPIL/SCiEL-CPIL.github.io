"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 6x6 into 3x3 blocks of 2x2.
    # Use schoolbook (rank 4) for each 2x2 block multiplication.
    # Total rank = 3*3*3*4 = 108.
    
    block_size = 2
    num_blocks = n // block_size # 3
    
    rank = num_blocks * num_blocks * num_blocks * (block_size * block_size * block_size)
    # rank = 3*3*3*8 = 216? No.
    # Schoolbook rank for 2x2 is 4.
    # Terms: 3 (intermediate) * 3 (row block) * 3 (col block) * 4 (local rank) = 108.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(num_blocks): # Row block of A, Row block of C
        for c in range(num_blocks): # Col block of B, Col block of C
            for k in range(num_blocks): # Intermediate block
                # We compute A[r,k] * B[k,c]
                # A[r,k] is 2x2, B[k,c] is 2x2.
                # Local schoolbook terms:
                # 1. A[0,0] * B[0,0] -> C[0,0]
                # 2. A[0,1] * B[1,0] -> C[0,0]
                # 3. A[1,0] * B[0,1] -> C[1,1]
                # 4. A[1,1] * B[1,1] -> C[1,1]
                
                # Global indices
                # A row: 2*r + u, A col: 2*k + v
                # B row: 2*k + v, B col: 2*c + w
                # C row: 2*r + u, C col: 2*c + w
                
                # Term 1: u=0, v=0, w=0
                u_row = 2 * r
                a_col = 2 * k
                U[u_row * m + a_col][col] = 1
                b_row = 2 * k
                b_col = 2 * c
                V[b_row * p + b_col][col] = 1
                c_row = 2 * r
                c_col = 2 * c
                W[c_col * n + c_row][col] = 1
                col += 1
                
                # Term 2: u=0, v=1, w=0
                u_row = 2 * r
                a_col = 2 * k + 1
                U[u_row * m + a_col][col] = 1
                b_row = 2 * k + 1
                b_col = 2 * c
                V[b_row * p + b_col][col] = 1
                c_row = 2 * r
                c_col = 2 * c
                W[c_col * n + c_row][col] = 1
                col += 1
                
                # Term 3: u=1, v=0, w=1
                u_row = 2 * r + 1
                a_col = 2 * k
                U[u_row * m + a_col][col] = 1
                b_row = 2 * k
                b_col = 2 * c + 1
                V[b_row * p + b_col][col] = 1
                c_row = 2 * r + 1
                c_col = 2 * c + 1
                W[c_col * n + c_row][col] = 1
                col += 1
                
                # Term 4: u=1, v=1, w=1
                u_row = 2 * r + 1
                a_col = 2 * k + 1
                U[u_row * m + a_col][col] = 1
                b_row = 2 * k + 1
                b_col = 2 * c + 1
                V[b_row * p + b_col][col] = 1
                c_row = 2 * r + 1
                c_col = 2 * c + 1
                W[c_col * n + c_row][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
