"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Strategy: Split p=6 into p1=4 and p2=2
    # Part 1: 4x4 x 4x4 using recursive Strassen (rank 49)
    # Part 2: 4x4 x 4x2 using standard multiplication (rank 32)
    
    rank_part1 = 49  # 7^2 for 4x4x4 recursive Strassen
    rank_part2 = 32  # 4*4*2 for standard 4x4x2
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x total_rank
    
    # Build 4x4x4 recursive Strassen decomposition (rank 49)
    # Recursive Strassen: split 4x4x4 into 8 blocks of 2x2x2
    # But with sharing: the 4x4x4 tensor = sum over 7 "meta-products" each of rank 7
    # Actually: 4x4x4 MM can be done with 49 scalar multiplications via recursive Strassen
    # The structure: for each of the 7 Strassen products at the top level,
    # we have a 2x2x2 block, and each of those uses 7 Strassen products at the bottom level
    # But we need to be careful about the indices
    
    # Let's build it explicitly using the recursive structure
    # Top level: split i into i1,i2 (each 2), j into j1,j2 (each 2), k into k1,k2 (each 2)
    # The 8 blocks are (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2),
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    
    # For 4x4x4, we use the fact that it's a Kronecker-like product of two 2x2x2 Strassen
    # The 4x4x4 MM tensor T[i,j,k] = 1 for all i,j,k
    # Can be decomposed as: sum_{a,b,c in {0,1}} T2[a,j1+k1,b] * ... 
    # Actually, let me use a direct construction
    
    # Recursive Strassen for 4x4x4:
    # Level 1: Apply Strassen to get 7 sub-problems of size 2x2x2
    # Level 2: Each 2x2x2 uses Strassen with 7 scalar products
    # Total: 7 * 7 = 49
    
    # The U, V, W matrices for 4x4x4 recursive Strassen:
    # U4: 16 x 49, V4: 16 x 49, W4: 16 x 49
    
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    # For each top-level Strassen product c1 in range(7):
    #   The U factor at top level gives coefficients for the 4x4 input blocks
    #   Then for each of those, we apply bottom-level Strassen
    # 
    # Top-level Strassen for 4x4x4 (treating 2x2 blocks as "elements"):
    # U_top: 4x7 for the 2x2 block indices
    # The 4x4 matrix is split into four 2x2 blocks: A00, A01, A10, A11
    # Similarly B: B00, B01, B10, B11 and C: C00, C01, C10, C11
    #
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix product, computed with Strassen (7 scalar products)
    #
    # So for the full 4x4x4 decomposition:
    # U4[i*m+j, r] where r = c1*7 + c2 for c1 in 0..6, c2 in 0..6
    # The top-level U gives which 2x2 blocks of A are involved
    # The bottom-level U gives which scalar elements within those 2x2 blocks
    
    # Top-level U coefficients for A blocks (rows are block indices 0,1,2,3 for A00,A01,A10,A11)
    U_top_blocks = [
        [1, 1, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 1, 0, 0],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 1],  # A11
    ]
    V_top_blocks = [
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, -1, 1, 0, 1], # B11
    ]
    W_top_blocks = [
        [1, 0, 0, 0, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]
    
    # Now build the full 4x4x4 factors
    for c1 in range(7):
        for c2 in range(7):
            r = c1 * 7 + c2
            # U4: rows (i,j) for i,j in 0..3
            # i = bi*2 + di, j = bj*2 + dj
            # Top-level block index for A: bi*2 + bj (A00=0, A01=1, A10=2, A11=3)
            # Bottom-level element within block: di*2 + dj
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    top_coeff = U_top_blocks[block_idx][c1]
                    if top_coeff == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            bottom_coeff = S_U[di * 2 + dj][c2]
                            U4[i * 4 + j][r] += top_coeff * bottom_coeff
            
            # V4: rows (j,k) for j,k in 0..3
            # j = bj*2 + dj, k = bk*2 + dk
            # Top-level block index for B: bj*2 + bk
            for bj in range(2):
                for bk in range(2):
                    block_idx = bj * 2 + bk
                    top_coeff = V_top_blocks[block_idx][c1]
                    if top_coeff == 0:
                        continue
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            bottom_coeff = S_V[dj * 2 + dk][c2]
                            V4[j * 4 + k][r] += top_coeff * bottom_coeff
            
            # W4: rows (k,i) for k,i in 0..3
            # k = bk*2 + dk, i = bi*2 + di
            # Top-level block index for C: bk*2 + bi
            for bk in range(2):
                for bi in range(2):
                    block_idx = bk * 2 + bi
                    top_coeff = W_top_blocks[block_idx][c1]
                    if top_coeff == 0:
                        continue
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            bottom_coeff = S_W[dk * 2 + di][c2]
                            W4[k * 4 + i][r] += top_coeff * bottom_coeff
    
    # Copy 4x4x4 part into main matrices (columns 0..48)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 6 + k][r] = V4[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][r] = W4[k * 4 + i][r]
    
    # Build 4x4x2 standard multiplication decomposition (rank 32)
    # This handles columns k=4,5 (the last 2 columns of the 4x6 output)
    # For each (i,j,k) with k in {4,5}, we need a term
    # Standard: T[i,j,k] = 1, so for each (i,j,k), one rank-1 term
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):  # k = 4, 5
                r = offset + (i * 4 + j) * 2 + (k - 4)
                U[i * 4 + j][r] = 1
                V[j * 6 + k][r] = 1
                W[k * 4 + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
