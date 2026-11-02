"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    # Rank calculation:
    # We tile the m x p grid (5x6) with 2x2 blocks (Strassen, rank 7) and 1x2 blocks (Standard, rank 4).
    # Tiling strategy:
    # Rows 0-1 (j=0,1): Cols 0-1, 2-3, 4-5 -> 3 blocks of 2x2 (Rank 7 each)
    # Rows 2-3 (j=2,3): Cols 0-1, 2-3, 4-5 -> 3 blocks of 2x2 (Rank 7 each)
    # Row 4 (j=4): Cols 0-1, 2-3, 4-5 -> 3 blocks of 1x2 (Rank 4 each)
    # Total Rank = 6 * 7 + 3 * 4 = 42 + 12 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2
    # u: (li, lj, val), v: (lj, lk, val), w: (li, lk, val)
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    def add_strassen_block(j_base, k_base):
        nonlocal col
        for term in strassen_terms:
            u_list, v_list, w_list = term
            for (li, lj, val) in u_list:
                U[li * m + (j_base + lj)][col] += val
            for (lj, lk, val) in v_list:
                V[(j_base + lj) * p + (k_base + lk)][col] += val
            for (li, lk, val) in w_list:
                W[(k_base + lk) * n + li][col] += val
            col += 1
            
    def add_standard_block_1x2(j, k_base):
        # Block size n=2, m=1, p=2
        # Rank 4
        nonlocal col
        for i in range(n):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + (k_base + k)][col] = 1
                W[(k_base + k) * n + i][col] = 1
                col += 1

    # Part 1: Rows 0-1 (j=0,1)
    # 3 blocks of 2x2
    for k_base in range(0, p, 2):
        add_strassen_block(0, k_base)
        
    # Part 2: Rows 2-3 (j=2,3)
    # 3 blocks of 2x2
    for k_base in range(0, p, 2):
        add_strassen_block(2, k_base)
        
    # Part 3: Row 4 (j=4)
    # 3 blocks of 1x2
    for k_base in range(0, p, 2):
        add_standard_block_1x2(4, k_base)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
