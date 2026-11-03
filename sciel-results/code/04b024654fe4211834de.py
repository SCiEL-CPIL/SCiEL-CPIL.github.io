"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Block 1 (i=0): 1x4x8. Split into 2x (1x2x2). Each 1x2x2 rank 4. Total 2*4*4 = 32.
    # Block 2 (i=1,2): 2x4x8. Split into 2x (2x2x2) x 4x (2x2x2). 
    # Wait, 2x4x8 split m into 2+2, p into 2+2+2+2.
    # 2 j-blocks, 4 k-blocks. Each 2x2x2 rank 7.
    # Total 2 * 4 * 7 = 56.
    # Total Rank = 32 + 56 = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 1x2x2 block (schoolbook)
    # i is fixed to 0
    # j_offset, k_offset
    def add_1x2x2(j_offset, k_offset):
        nonlocal col
        # A: 1x2 (indices 0, j_offset..j_offset+1)
        # B: 2x2 (indices j_offset..j_offset+1, k_offset..k_offset+1)
        # C: 1x2 (indices 0, k_offset..k_offset+1)
        # C_k = A_0 B_0k + A_1 B_1k
        for j_local in range(2):
            for k_local in range(2):
                # U[i*m + j, c] = 1
                # V[j*p + k, c] = 1
                # W[k*n + i, c] = 1
                i = 0
                j = j_offset + j_local
                k = k_offset + k_local
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Helper to add a 2x2x2 block (Strassen)
    # i_offset (1 or 2), j_offset, k_offset
    def add_2x2x2(i_offset, j_offset, k_offset):
        nonlocal col
        # Local indices 0,1 for i, j, k
        # Global i = i_offset + i_local
        # Global j = j_offset + j_local
        # Global k = k_offset + k_local
        
        # Strassen factors:
        # z1 = a00 * (b01 + b11)
        # z2 = (a00 + a01) * b00
        # z3 = (a01 + a11) * b11
        # z4 = a11 * (b10 - b00)
        # z5 = a10 * (b00 + b11)
        # z6 = (a00 - a10) * (b10 + b11)
        # z7 = (a01 - a11) * (b00 + b10)
        
        # Coefficients for U (A), V (B), W (C)
        # U rows: 0, 1 (local)
        # V cols: 0, 1 (local)
        # W cols: 0, 1 (local)
        
        # We will add 7 columns to U, V, W
        
        # z1: U[0]=1, V[0]=0, V[1]=1 (b01+b11 is sum of B[0,1] and B[1,1])
        # Wait, V is indexed by [j, k].
        # b01 is B[0,1] -> V[0,1]
        # b11 is B[1,1] -> V[1,1]
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: a00 * (b01 + b11)
        U[i_offset * m + j_offset + 0][col] = 1
        V[(j_offset + 0) * p + (k_offset + 1)][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # C00 += -z1? No, C00 = z5+z7-z3-z4. C01 = z1+z4. C10 = z2+z3. C11 = z5-z1+z6.
        # W[k*n + i, c]
        # C01 gets z1. C01 is W[1*n + 0, c] (k=1, i=0 local)
        W[(k_offset + 1) * n + (i_offset + 0)][col] = 1
        col += 1
        
        # Term 2: (a00 + a01) * b00
        U[i_offset * m + j_offset + 0][col] = 1
        U[i_offset * m + j_offset + 1][col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col] = 1
        # C10 gets z2. C10 is W[0*n + 1, c] (k=0, i=1 local)
        W[(k_offset + 0) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # Term 3: (a01 + a11) * b11
        U[i_offset * m + j_offset + 1][col] = 1
        U[i_offset * m + j_offset + 1][col] = 1 # a11
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # C10 gets z3.
        W[(k_offset + 0) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # Term 4: a11 * (b10 - b00)
        U[i_offset * m + j_offset + 1][col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col] = -1
        # C01 gets z4. C00 gets -z4.
        W[(k_offset + 1) * n + (i_offset + 0)][col] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][col] = -1
        col += 1
        
        # Term 5: a10 * (b00 + b11)
        U[i_offset * m + j_offset + 0][col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # C00 gets z5. C11 gets z5.
        W[(k_offset + 0) * n + (i_offset + 0)][col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # Term 6: (a00 - a10) * (b10 + b11)
        U[i_offset * m + j_offset + 0][col] = 1
        U[i_offset * m + j_offset + 0][col] = -1 # a10
        V[(j_offset + 1) * p + (k_offset + 0)][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # C11 gets z6.
        W[(k_offset + 1) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # Term 7: (a01 - a11) * (b00 + b10)
        U[i_offset * m + j_offset + 1][col] = 1
        U[i_offset * m + j_offset + 1][col] = -1 # a11
        V[(j_offset + 0) * p + (k_offset + 0)][col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][col] = 1
        # C00 gets z7.
        W[(k_offset + 0) * n + (i_offset + 0)][col] = 1
        col += 1

    # Execute Block 1: i=0
    # j blocks: 0-1, 2-3
    # k blocks: 0-1, 2-3, 4-5, 6-7
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_1x2x2(j_off, k_off)
            
    # Execute Block 2: i=1,2
    # j blocks: 0-1, 2-3
    # k blocks: 0-1, 2-3, 4-5, 6-7
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_2x2x2(1, j_off, k_off)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
