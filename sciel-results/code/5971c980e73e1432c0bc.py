"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Split p=8 into two p=4 blocks.
    # For each p=4 block:
    #   Split m=3 into m=2 and m=1.
    #   m=2, p=4: Split p=4 into two p=2. Use Strassen (rank 7) for each 2x2x2. Total 14.
    #   m=1, p=4: Use Schoolbook (rank 8) for 2x1x4. Total 8.
    #   Total per p=4 block: 22.
    # Total rank: 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Strassen 7 multiplications for 2x2x2
        # Indices:
        # U: (i, j) -> idx = i*m + j
        # V: (j, k) -> idx = j*p + k
        # W: (k, i) -> idx = k*n + i
        
        # 1
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 2
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # 3
        U[(i_off)*m + (j_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 4
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off)][col] += -1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        col += 1
        
        # 5
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        col += 1
        
        # 6
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off)][col] += -1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 7
        U[(i_off)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off)][col] = 1
                V[(j_off)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Process first p=4 block (k=0..3)
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 0) # k=0,1
    add_strassen_2x2x2(0, 0, 2) # k=2,3
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 0) # k=0..3

    # Process second p=4 block (k=4..7)
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 4) # k=4,5
    add_strassen_2x2x2(0, 0, 6) # k=6,7
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 4) # k=4..7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
