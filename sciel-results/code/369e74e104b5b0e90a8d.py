"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    # A (3x4) = [A1 (3x2), A2 (3x2)]
    # B (4x5) = [[B11 (2x2), B12 (2x2), B13 (2x1)],
    #            [B21 (2x2), B22 (2x2), B23 (2x1)]]
    # C = A1*B1 + A2*B2
    # B1 = [B11, B12, B13]
    # B2 = [B21, B22, B23]
    
    # We will build U, V, W
    # U: (n*m, rank) = (12, 42)
    # V: (m*p, rank) = (20, 42)
    # W: (p*n, rank) = (15, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add 2x2x2 Strassen
    def add_strassen_2x2x2(u_off, v_off, w_off, u_map, v_map, w_map):
        nonlocal col
        # u_map: list of 4 indices for A (2x2)
        # v_map: list of 4 indices for B (2x2)
        # w_map: list of 4 indices for C (2x2)
        
        # Term 1: (a+d)(e+h)
        U[u_map[0]][col] = 1; U[u_map[3]][col] = 1
        V[v_map[0]][col] = 1; V[v_map[3]][col] = 1
        W[w_map[0]][col] = 1; W[w_map[3]][col] = 1
        col += 1
        
        # Term 2: (c+d)e
        U[u_map[2]][col] = 1; U[u_map[3]][col] = 1
        V[v_map[0]][col] = 1
        W[w_map[2]][col] = 1; W[w_map[3]][col] = -1
        col += 1
        
        # Term 3: a(f-h)
        U[u_map[0]][col] = 1
        V[v_map[1]][col] = 1; V[v_map[3]][col] = -1
        W[w_map[1]][col] = 1; W[w_map[3]][col] = 1
        col += 1
        
        # Term 4: d(g-e)
        U[u_map[3]][col] = 1
        V[v_map[2]][col] = 1; V[v_map[0]][col] = -1
        W[w_map[0]][col] = 1; W[w_map[2]][col] = 1
        col += 1
        
        # Term 5: (a+b)h
        U[u_map[0]][col] = 1; U[u_map[1]][col] = 1
        V[v_map[3]][col] = 1
        W[w_map[0]][col] = -1; W[w_map[1]][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f)
        U[u_map[2]][col] = 1; U[u_map[0]][col] = -1
        V[v_map[0]][col] = 1; V[v_map[1]][col] = 1
        W[w_map[3]][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h)
        U[u_map[1]][col] = 1; U[u_map[3]][col] = -1
        V[v_map[2]][col] = 1; V[v_map[3]][col] = 1
        W[w_map[0]][col] = 1
        col += 1

    # Helper to add 1x2x2 (rank 2)
    # A (1x2) * B (2x2) -> C (1x2)
    # C_0 = A_0 B_0 + A_1 B_1
    # C_1 = A_0 B_2 + A_1 B_3
    # B indices: 0,1,2,3
    def add_1x2x2(u_idx, v_off, w_off, w_map):
        nonlocal col
        # Term 1: A_0 * B_0 -> C_0, A_0 * B_2 -> C_1
        U[u_idx][col] = 1
        V[v_off + 0][col] = 1; V[v_off + 2][col] = 1
        W[w_map[0]][col] = 1; W[w_map[1]][col] = 1
        col += 1
        
        # Term 2: A_1 * B_1 -> C_0, A_1 * B_3 -> C_1
        U[u_idx][col] = 1
        V[v_off + 1][col] = 1; V[v_off + 3][col] = 1
        W[w_map[0]][col] = 1; W[w_map[1]][col] = 1
        col += 1

    # Helper to add 3x2x2 (rank 9)
    # A (3x2) * B (2x2) -> C (3x2)
    # Split A into A_top (2x2) and A_bot (1x2)
    def add_3x2x2(u_off, v_off, w_off, u_map, v_map, w_map):
        # u_map: 6 indices for A (3x2)
        # v_map: 4 indices for B (2x2)
        # w_map: 6 indices for C (3x2)
        
        # Top 2x2x2
        u_top = u_map[0:4]
        w_top = w_map[0:4]
        add_strassen_2x2x2(0, 0, 0, u_top, v_map, w_top)
        # Note: add_strassen uses local indices 0-3. 
        # We need to map them to global u, v, w.
        # My helper above takes u_off etc. but I changed it to take maps.
        # Let's refactor add_strassen to take maps directly.
        
    # Refactored helpers taking maps
    
    def add_strassen_map(u_map, v_map, w_map):
        nonlocal col
        # u_map, v_map, w_map are lists of 4 global indices
        # Term 1
        U[u_map[0]][col] = 1; U[u_map[3]][col] = 1
        V[v_map[0]][col] = 1; V[v_map[3]][col] = 1
        W[w_map[0]][col] = 1; W[w_map[3]][col] = 1
        col += 1
        # Term 2
        U[u_map[2]][col] = 1; U[u_map[3]][col] = 1
        V[v_map[0]][col] = 1
        W[w_map[2]][col] = 1; W[w_map[3]][col] = -1
        col += 1
        # Term 3
        U[u_map[0]][col] = 1
        V[v_map[1]][col] = 1; V[v_map[3]][col] = -1
        W[w_map[1]][col] = 1; W[w_map[3]][col] = 1
        col += 1
        # Term 4
        U[u_map[3]][col] = 1
        V[v_map[2]][col] = 1; V[v_map[0]][col] = -1
        W[w_map[0]][col] = 1; W[w_map[2]][col] = 1
        col += 1
        # Term 5
        U[u_map[0]][col] = 1; U[u_map[1]][col] = 1
        V[v_map[3]][col] = 1
        W[w_map[0]][col] = -1; W[w_map[1]][col] = 1
        col += 1
        # Term 6
        U[u_map[2]][col] = 1; U[u_map[0]][col] = -1
        V[v_map[0]][col] = 1; V[v_map[1]][col] = 1
        W[w_map[3]][col] = 1
        col += 1
        # Term 7
        U[u_map[1]][col] = 1; U[u_map[3]][col] = -1
        V[v_map[2]][col] = 1; V[v_map[3]][col] = 1
        W[w_map[0]][col] = 1
        col += 1

    def add_1x2x2_map(u_idx, v_map, w_map):
        nonlocal col
        # v_map: 4 indices for B (2x2)
        # w_map: 2 indices for C (1x2)
        # Term 1: A_0
        U[u_idx][col] = 1
        V[v_map[0]][col] = 1; V[v_map[2]][col] = 1
        W[w_map[0]][col] = 1; W[w_map[1]][col] = 1
        col += 1
        # Term 2: A_1
        U[u_idx][col] = 1
        V[v_map[1]][col] = 1; V[v_map[3]][col] = 1
        W[w_map[0]][col] = 1; W[w_map[1]][col] = 1
        col += 1

    def add_3x2x2_map(u_map, v_map, w_map):
        # u_map: 6 indices (3x2)
        # v_map: 4 indices (2x2)
        # w_map: 6 indices (3x2)
        # Top 2x2
        u_top = u_map[0:4]
        w_top = w_map[0:4]
        add_strassen_map(u_top, v_map, w_top)
        # Bot 1x2
        u_bot = u_map[4]
        w_bot = w_map[4:6]
        add_1x2x2_map(u_bot, v_map, w_bot)

    def add_3x2x1_map(u_map, v_map, w_map):
        # u_map: 6 indices (3x2)
        # v_map: 2 indices (2x1)
        # w_map: 3 indices (3x1)
        # Rank 3
        # C_0 = A_00 B_0 + A_01 B_1
        # C_1 = A_10 B_0 + A_11 B_1
        # C_2 = A_20 B_0 + A_21 B_1
        nonlocal col
        # Term 1: B_0
        # A_00, A_10, A_20
        U[u_map[0]][col] = 1
        U[u_map[2]][col] = 1
        U[u_map[4]][col] = 1
        V[v_map[0]][col] = 1
        W[w_map[0]][col] = 1
        W[w_map[1]][col] = 1
        W[w_map[2]][col] = 1
        col += 1
        # Term 2: B_1
        # A_01, A_11, A_21
        U[u_map[1]][col] = 1
        U[u_map[3]][col] = 1
        U[u_map[5]][col] = 1
        V[v_map[1]][col] = 1
        W[w_map[0]][col] = 1
        W[w_map[1]][col] = 1
        W[w_map[2]][col] = 1
        col += 1

    # Define index maps
    # A is 3x4. Indices: i*4 + j. i=0..2, j=0..3.
    # A1: j=0,1. A2: j=2,3.
    # A1 indices: 0,1, 4,5, 8,9
    # A2 indices: 2,3, 6,7, 10,11
    
    # B is 4x5. Indices: j*5 + k. j=0..3, k=0..4.
    # B1: j=0,1. B2: j=2,3.
    # B1:
    #   j=0: 0,1,2,3,4
    #   j=1: 5,6,7,8,9
    # B2:
    #   j=2: 10,11,12,13,14
    #   j=3: 15,16,17,18,19
    
    # C is 3x5. Indices: k*3 + i. k=0..4, i=0..2.
    # C1 (from A1 B1) and C2 (from A2 B2) add up.
    # So W indices are the same for both blocks.
    
    # Block 1: A1 (3x2) * B1 (2x5)
    # A1 u_map: [0,1, 4,5, 8,9]
    # B1 v_map:
    #   B11 (2x2): j=0,1; k=0,1 -> 0,1, 5,6
    #   B12 (2x2): j=0,1; k=2,3 -> 2,3, 7,8
    #   B13 (2x1): j=0,1; k=4   -> 4, 9
    # C1 w_map (3x5):
    #   k=0: 0,1,2
    #   k=1: 3,4,5
    #   k=2: 6,7,8
    #   k=3: 9,10,11
    #   k=4: 12,13,14
    
    u_A1 = [0, 1, 4, 5, 8, 9]
    v_B11 = [0, 1, 5, 6]
    v_B12 = [2, 3, 7, 8]
    v_B13 = [4, 9]
    
    # W maps for C1
    # C1 is 3x5.
    # k=0: i=0,1,2 -> 0,1,2
    # k=1: i=0,1,2 -> 3,4,5
    # k=2: i=0,1,2 -> 6,7,8
    # k=3: i=0,1,2 -> 9,10,11
    # k=4: i=0,1,2 -> 12,13,14
    
    # For 3x2x2 (k=0,1)
    w_C1_01 = [0,1, 3,4, 6,7] # i=0,1,2 for k=0,1
    # Wait, w_map for 3x2x2 expects 6 indices.
    # k=0: i=0,1,2 -> 0,1,2
    # k=1: i=0,1,2 -> 3,4,5
    # So w_map = [0,1,2, 3,4,5]
    
    w_C1_k01 = [0, 1, 2, 3, 4, 5]
    w_C1_k23 = [6, 7, 8, 9, 10, 11]
    w_C1_k4   = [12, 13, 14]
    
    # Add A1 B11 (3x2x2)
    add_3x2x2_map(u_A1, v_B11, w_C1_k01)
    # Add A1 B12 (3x2x2)
    add_3x2x2_map(u_A1, v_B12, w_C1_k23)
    # Add A1 B13 (3x2x1)
    add_3x2x1_map(u_A1, v_B13, w_C1_k4)
    
    # Block 2: A2 (3x2) * B2 (2x5)
    u_A2 = [2, 3, 6, 7, 10, 11]
    v_B21 = [10, 11, 15, 16]
    v_B22 = [12, 13, 17, 18]
    v_B23 = [14, 19]
    
    # W maps are the same as C1 because they add up
    # Add A2 B21 (3x2x2)
    add_3x2x2_map(u_A2, v_B21, w_C1_k01)
    # Add A2 B22 (3x2x2)
    add_3x2x2_map(u_A2, v_B22, w_C1_k23)
    # Add A2 B23 (3x2x1)
    add_3x2x1_map(u_A2, v_B23, w_C1_k4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
