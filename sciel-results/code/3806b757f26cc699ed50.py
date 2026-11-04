"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition: 5 = 4 + 1
    # A = [A44 A41]
    #     [A14 A11]
    # B = [B44 B41]
    #     [B14 B11]
    # C = [C44 C41]
    #     [C14 C11]
    
    # C44 = A44*B44 + A41*B14
    # C41 = A44*B41 + A41*B11
    # C14 = A14*B44 + A11*B14
    # C11 = A14*B41 + A11*B11
    
    # We will construct the U, V, W matrices for each term.
    # Term 1: A44*B44 (4x4 * 4x4 -> 4x4)
    # Schoolbook rank 16
    # Term 2: A41*B14 (4x1 * 1x4 -> 4x4)
    # Rank 4
    # Term 3: A44*B41 (4x4 * 4x1 -> 4x1)
    # Rank 16
    # Term 4: A41*B11 (4x1 * 1x1 -> 4x1)
    # Rank 4
    # Term 5: A14*B44 (1x4 * 4x4 -> 1x4)
    # Rank 16
    # Term 6: A11*B14 (1x1 * 1x4 -> 1x4)
    # Rank 4
    # Term 7: A14*B41 (1x4 * 4x1 -> 1x1)
    # Rank 4
    # Term 8: A11*B11 (1x1 * 1x1 -> 1x1)
    # Rank 1
    
    # Total Rank = 16 + 4 + 16 + 4 + 16 + 4 + 4 + 1 = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i_start, i_end, j_start, j_end, k_start, k_end, u_val, v_val, w_val):
        nonlocal col
        # This helper is too generic, let's just write the loops.
        pass

    # Helper to add a schoolbook term for A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
    # Note: The indices in U, V, W are global.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_schoolbook(i1, i2, j1, j2, k1, k2):
        nonlocal col
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1

    # 1. A44 * B44 -> C44
    # i in 0..3, j in 0..3, k in 0..3
    add_schoolbook(0, 4, 0, 4, 0, 4)
    
    # 2. A41 * B14 -> C44
    # A41: i in 0..3, j=4. B14: j=4, k in 0..3.
    # C44: i in 0..3, k in 0..3.
    # Term: A[i,4] * B[4,k] -> C[i,k]
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1.0
            V[4 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    # 3. A44 * B41 -> C41
    # A44: i in 0..3, j in 0..3. B41: j in 0..3, k=4.
    # C41: i in 0..3, k=4.
    add_schoolbook(0, 4, 0, 4, 4, 5)
    
    # 4. A41 * B11 -> C41
    # A41: i in 0..3, j=4. B11: j=4, k=4.
    # C41: i in 0..3, k=4.
    for i in range(4):
        U[i * m + 4][col] = 1.0
        V[4 * p + 4][col] = 1.0
        W[4 * n + i][col] = 1.0
        col += 1
        
    # 5. A14 * B44 -> C14
    # A14: i=4, j in 0..3. B44: j in 0..3, k in 0..3.
    # C14: i=4, k in 0..3.
    add_schoolbook(4, 5, 0, 4, 0, 4)
    
    # 6. A11 * B14 -> C14
    # A11: i=4, j=4. B14: j=4, k in 0..3.
    # C14: i=4, k in 0..3.
    for k in range(4):
        U[4 * m + 4][col] = 1.0
        V[4 * p + k][col] = 1.0
        W[k * n + 4][col] = 1.0
        col += 1
        
    # 7. A14 * B41 -> C11
    # A14: i=4, j in 0..3. B41: j in 0..3, k=4.
    # C11: i=4, k=4.
    for j in range(4):
        U[4 * m + j][col] = 1.0
        V[j * p + 4][col] = 1.0
        W[4 * n + 4][col] = 1.0
        col += 1
        
    # 8. A11 * B11 -> C11
    # A11: i=4, j=4. B11: j=4, k=4.
    # C11: i=4, k=4.
    U[4 * m + 4][col] = 1.0
    V[4 * p + 4][col] = 1.0
    W[4 * n + 4][col] = 1.0
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
