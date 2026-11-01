"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Strassen decomposition for (2,2,2)
    # U[2i+j, r], V[2j+k, r], W[2k+i, r]
    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0],   # a00
        [0, 0, 0, 0, 0, 0, 0],   # a01
        [0, 1, 0, 0, 0, -1, 1],  # a10
        [1, 1, 0, 1, 1, 0, -1]   # a11
    ]
    strassen_V = [
        [1, 1, 0, 1, 0, 0, 1],   # b00
        [0, 0, 0, 0, 0, 1, 1],   # b01
        [0, 0, -1, -1, 1, 0, 0], # b10
        [1, 0, 1, 0, 0, 1, 0]    # b11
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # c00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # c01 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],   # c10 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0]   # c11 (k=1,i=1)
    ]
    
    # Schoolbook for (2,2,2) - rank 4
    schoolbook_U = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1]   # a11
    ]
    schoolbook_V = [
        [1, 0, 0, 0],  # b00
        [0, 1, 0, 0],  # b01
        [0, 0, 1, 0],  # b10
        [0, 0, 0, 1]   # b11
    ]
    schoolbook_W = [
        [1, 0, 0, 0],  # c00
        [0, 1, 0, 0],  # c01
        [0, 0, 1, 0],  # c10
        [0, 0, 0, 1]   # c11
    ]
    
    # Build the full decomposition
    # Columns 0-7: 4 pairs, each with Strassen (rank 7)
    # Column 8: schoolbook (rank 4)
    total_rank = 4 * 7 + 4  # = 32
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * total_rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    # 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        for r in range(7):
            # U: same for all pairs (shared A factors)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = strassen_U[i * m + j][r]
            # V: for j=0, k=k1 and k=k2; for j=1, k=k1 and k=k2
            for j in range(m):
                # Map local k (0 or 1) to global k
                for local_k in range(2):
                    global_k = k1 + local_k
                    V[j * p + global_k][col] = strassen_V[j * 2 + local_k][r]
            # W: for k=k1 and k=k2, i=0 and i=1
            for local_k in range(2):
                global_k = k1 + local_k
                for i in range(n):
                    W[global_k * n + i][col] = strassen_W[local_k * n + i][r]
            col += 1
    
    # Column 8: schoolbook
    for r in range(4):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = schoolbook_U[i * m + j][r]
        for j in range(m):
            V[j * p + 8][col] = schoolbook_V[j * 2][r]  # local k=0 for single column
        for i in range(n):
            W[8 * n + i][col] = schoolbook_W[0 * n + i][r]  # local k=0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
