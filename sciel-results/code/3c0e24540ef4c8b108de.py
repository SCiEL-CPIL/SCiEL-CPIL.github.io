"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition:
    # Split m=5 into m1=1 and m2=4.
    # Part 1: 2x1x6 block. Rank = 2*6 = 12.
    # Part 2: 2x4x6 block. Split m2=4 into two 2x2x6 blocks.
    # Each 2x2x6 block has rank 21 (3 Strassen instances of rank 7).
    # Total Rank = 12 + 21 + 21 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for i, j, val in u_indices:
            U[i * m + j][col] += val
        for j, k, val in v_indices:
            V[j * p + k][col] += val
        for i, k, val in w_indices:
            W[k * n + i][col] += val
        col += 1

    # Part 1: j=0 (2x1x6)
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            add_term([(i, 0, 1)], [(0, k, 1)], [(i, k, 1)])

    # Part 2: j in 1..4 (2x4x6)
    # Split into two 2x2 blocks: j in [1,2] and j in [3,4]
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        
        # For each 2x2x2 sub-block in the 6 columns (3 blocks of 2 cols)
        for k_block in range(3):
            k_base = k_block * 2
            
            # Strassen terms for 2x2x2
            # Local indices:
            # A: (0,0), (0,1), (1,0), (1,1)
            # B: (0,0), (0,1), (1,0), (1,1)
            # C: (0,0), (0,1), (1,0), (1,1)
            
            # M1: (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
            add_term(
                [(0, 0, 1), (1, 1, 1)],
                [(0, 0, 1), (1, 1, 1)],
                [(0, 0, 1), (1, 1, 1)]
            )
            
            # M2: (A10 + A11)B00 -> C10 += 1, C11 -= 1
            add_term(
                [(1, 0, 1), (1, 1, 1)],
                [(0, 0, 1)],
                [(1, 0, 1), (1, 1, -1)]
            )
            
            # M3: A00(B01 - B11) -> C01 += 1, C11 += 1
            add_term(
                [(0, 0, 1)],
                [(0, 1, 1), (1, 1, -1)],
                [(0, 1, 1), (1, 1, 1)]
            )
            
            # M4: A11(B10 - B00) -> C00 += 1, C10 += 1
            add_term(
                [(1, 1, 1)],
                [(1, 0, 1), (0, 0, -1)],
                [(0, 0, 1), (1, 0, 1)]
            )
            
            # M5: (A00 + A01)B11 -> C00 -= 1, C01 += 1
            add_term(
                [(0, 0, 1), (0, 1, 1)],
                [(1, 1, 1)],
                [(0, 0, -1), (0, 1, 1)]
            )
            
            # M6: (A10 - A00)(B00 + B01) -> C11 += 1
            add_term(
                [(1, 0, 1), (0, 0, -1)],
                [(0, 0, 1), (0, 1, 1)],
                [(1, 1, 1)]
            )
            
            # M7: (A01 - A11)(B10 + B11) -> C00 += 1
            add_term(
                [(0, 1, 1), (1, 1, -1)],
                [(1, 0, 1), (1, 1, 1)],
                [(0, 0, 1)]
            )
            
            # Map local indices to global indices for the next iteration?
            # No, add_term uses local indices relative to the block?
            # Wait, the add_term above uses global indices directly if we pass them.
            # But I passed local indices (0,1).
            # I need to shift them.
            # Let's refactor add_term to accept offsets.
            
    # The above logic in add_term is flawed because it uses local indices 0,1 
    # but U, V, W are global.
    # I will rewrite the loop to calculate global indices explicitly.
    
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1..4
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        for k_block in range(3):
            k_base = k_block * 2
            
            # Strassen Terms
            # M1
            U[0 * m + (j_base + 0)][col] += 1
            U[1 * m + (j_base + 1)][col] += 1
            V[(j_base + 0) * p + (k_base + 0)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[(k_base + 0) * n + 0][col] += 1
            W[(k_base + 1) * n + 1][col] += 1
            col += 1
            
            # M2
            U[1 * m + (j_base + 0)][col] += 1
            U[1 * m + (j_base + 1)][col] += 1
            V[(j_base + 0) * p + (k_base + 0)][col] += 1
            W[(k_base + 0) * n + 1][col] += 1
            W[(k_base + 1) * n + 1][col] -= 1
            col += 1
            
            # M3
            U[0 * m + (j_base + 0)][col] += 1
            V[(j_base + 0) * p + (k_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] -= 1
            W[(k_base + 1) * n + 0][col] += 1
            W[(k_base + 1) * n + 1][col] += 1
            col += 1
            
            # M4
            U[1 * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 0)][col] += 1
            V[(j_base + 0) * p + (k_base + 0)][col] -= 1
            W[(k_base + 0) * n + 0][col] += 1
            W[(k_base + 0) * n + 1][col] += 1
            col += 1
            
            # M5
            U[0 * m + (j_base + 0)][col] += 1
            U[0 * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[(k_base + 0) * n + 0][col] -= 1
            W[(k_base + 1) * n + 0][col] += 1
            col += 1
            
            # M6
            U[1 * m + (j_base + 0)][col] += 1
            U[0 * m + (j_base + 0)][col] -= 1
            V[(j_base + 0) * p + (k_base + 0)][col] += 1
            V[(j_base + 0) * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + 1][col] += 1
            col += 1
            
            # M7
            U[0 * m + (j_base + 1)][col] += 1
            U[1 * m + (j_base + 1)][col] -= 1
            V[(j_base + 1) * p + (k_base + 0)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[(k_base + 0) * n + 0][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
