"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: 5 independent 2x2 blocks, each rank 7, total rank 35
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks
    rank = 7 * num_blocks  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for each of the 7 terms
    # U: [A11, A12, A21, A22] contributions
    # V: [B11, B12, B21, B22] contributions for the block
    # W: [C11, C12, C21, C22] contributions
    strassen_U = [
        [1, 0, 0, 0],   # M1: A11
        [1, 1, 0, 0],   # M2: A11+A12
        [0, 0, 1, 1],   # M3: A21+A22
        [0, 0, 0, 1],   # M4: A22
        [1, 0, 0, 1],   # M5: A11+A22
        [0, 1, 0, -1],  # M6: A12-A22
        [1, 0, -1, 0],  # M7: A11-A21
    ]
    strassen_V = [
        [0, 1, 0, -1],  # M1: B12-B22
        [0, 0, 0, 1],   # M2: B22
        [1, 0, 0, 0],   # M3: B11
        [-1, 0, 1, 0],  # M4: B21-B11
        [1, 0, 0, 1],   # M5: B11+B22
        [0, 0, 1, 1],   # M6: B21+B22
        [1, 1, 0, 0],   # M7: B11+B12
    ]
    strassen_W = [
        [0, 0, 0, 0],   # M1 contributes to C12 only? Let me recheck
        # Actually: C11 = M5+M4-M2+M6, C12 = M1+M2, C21 = M3+M4, C22 = M5+M1-M3+M7
        # So W for each term:
        # M1: C12 += 1, C22 += 1
        # M2: C11 -= 1, C12 += 1
        # M3: C21 += 1, C22 -= 1
        # M4: C11 += 1, C21 += 1
        # M5: C11 += 1, C22 += 1
        # M6: C11 += 1
        # M7: C22 += 1
    ]
    
    # Let me define W properly: [C11, C12, C21, C22] for each term
    strassen_W = [
        [0, 1, 0, 1],   # M1: C12 += 1, C22 += 1
        [-1, 1, 0, 0],  # M2: C11 -= 1, C12 += 1
        [0, 0, 1, -1],  # M3: C21 += 1, C22 -= 1
        [1, 0, 1, 0],   # M4: C11 += 1, C21 += 1
        [1, 0, 0, 1],   # M5: C11 += 1, C22 += 1
        [1, 0, 0, 0],   # M6: C11 += 1
        [0, 0, 0, 1],   # M7: C22 += 1
    ]
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            
            # Fill U (4 entries: A11, A12, A21, A22)
            for a_idx in range(4):
                U[a_idx][col] = strassen_U[t][a_idx]
            
            # Fill V (20 entries, but only 4 relevant for this block)
            # B[0,2b]=2b, B[0,2b+1]=2b+1, B[1,2b]=10+2b, B[1,2b+1]=10+2b+1
            B_indices = [2*b, 2*b+1, 10+2*b, 10+2*b+1]
            for b_idx in range(4):
                V[B_indices[b_idx]][col] = strassen_V[t][b_idx]
            
            # Fill W (20 entries, but only 4 relevant for this block)
            # C[0,2b]=4b, C[0,2b+1]=4b+2, C[1,2b]=4b+1, C[1,2b+1]=4b+3
            C_indices = [4*b, 4*b+2, 4*b+1, 4*b+3]
            for c_idx in range(4):
                W[C_indices[c_idx]][col] = strassen_W[t][c_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
