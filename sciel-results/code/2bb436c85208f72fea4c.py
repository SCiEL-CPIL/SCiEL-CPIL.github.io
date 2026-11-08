import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 3x4x6
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x6) into B1 (2x6) and B2 (2x6)
    # C = A1 B1 + A2 B2
    
    # We need factors for 3x2x6
    # Split B1 (2x6) into B1a (2x3) and B1b (2x3)
    # C1 = A1 B1a + A1 B1b
    # We need factors for 3x2x3
    
    # Split A1 (3x2) into A1a (2x2) and A1b (1x2)
    # Split B1a (2x3) into B1aa (2x2) and B1ab (2x1)
    # C1a = A1a B1aa + A1a B1ab + A1b B1aa + A1b B1ab
    
    # Base cases:
    # 2x2x2: Rank 7 (Strassen)
    # 2x2x1: Rank 4 (Direct)
    # 1x2x2: Rank 4 (Direct)
    # 1x2x1: Rank 2 (Direct)
    
    # Helper to add terms to U, V, W
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    U = [[0] * 68 for _ in range(n * m)]
    V = [[0] * 68 for _ in range(m * p)]
    W = [[0] * 68 for _ in range(p * n)]
    
    col = 0
    
    # Function to add a block decomposition
    # i_offset, j_offset, k_offset are offsets in the global indices
    # n_b, m_b, p_b are dimensions of the block
    # U_b, V_b, W_b are the factors for the block
    
    def add_block(i_off, j_off, k_off, n_b, m_b, p_b, U_b, V_b, W_b, rank_b):
        nonlocal col
        for r in range(rank_b):
            for i in range(n_b):
                for j in range(m_b):
                    if U_b[i * m_b + j][r] != 0:
                        U[(i + i_off) * m + (j + j_off)][col] = U_b[i * m_b + j][r]
            for j in range(m_b):
                for k in range(p_b):
                    if V_b[j * p_b + k][r] != 0:
                        V[(j + j_off) * p + (k + k_off)][col] = V_b[j * p_b + k][r]
            for k in range(p_b):
                for i in range(n_b):
                    if W_b[k * p_b + i][r] != 0:
                        W[(k + k_off) * n + (i + i_off)][col] = W_b[k * p_b + i][r]
            col += 1

    # Define Base Factors
    
    # 2x2x2 Strassen
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # M1
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[1][0] = 1; W_222[2][0] = 1; W_222[3][0] = 1 # Placeholder, need exact W
    
    # Correct Strassen W:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Reset W_222
    W_222 = [[0]*7 for _ in range(4)]
    # M1
    W_222[0][0] = 1 # C00
    W_222[3][0] = 1 # C11
    # M2
    W_222[2][1] = 1 # C10
    W_222[3][1] = -1 # C11
    # M3
    W_222[1][2] = 1 # C01
    W_222[3][2] = 1 # C11
    # M4
    W_222[0][3] = 1 # C00
    W_222[2][3] = 1 # C10
    # M5
    W_222[0][4] = -1 # C00
    W_222[1][4] = 1 # C01
    # M6
    W_222[3][5] = 1 # C11
    # M7
    W_222[0][6] = 1 # C00
    W_222[3][6] = 1 # C11
    
    # U and V for Strassen
    U_222 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 0]   # A11
    ]
    V_222 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 0, 0],  # B01
        [0, 0, -1, 0, 0, 0, 1], # B10
        [1, 0, 0, 1, 1, 0, 1]   # B11
    ]
    
    # 2x2x1 Direct (Rank 4)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    U_221 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_221 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_221 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # 1x2x2 Direct (Rank 4)
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    U_122 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V_122 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W_122 = [
        [1, 0, 0, 0],
        [0, 1, 1, 0] # Wait, W indices are k*n+i. n=1.
    ]
    # W for 1x2x2:
    # Term 0: A0 B00 -> C0. W[0, 0] = 1.
    # Term 1: A1 B10 -> C0. W[0, 1] = 1.
    # Term 2: A0 B01 -> C1. W[1, 0] = 1.
    # Term 3: A1 B11 -> C1. W[1, 1] = 1.
    W_122 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # 1x2x1 Direct (Rank 2)
    # C0 = A0 B0 + A1 B1
    U_121 = [
        [1, 0],
        [0, 1]
    ]
    V_121 = [
        [1, 0],
        [0, 1]
    ]
    W_121 = [
        [1, 1]
    ]
    
    # Assemble 3x2x3
    # A1a (2x2) at (0,0), A1b (1x2) at (2,0)
    # B1aa (2x2) at (0,0), B1ab (2x1) at (0,2)
    
    U_323 = [[0]*17 for _ in range(6)]
    V_323 = [[0]*17 for _ in range(6)]
    W_323 = [[0]*17 for _ in range(9)]
    
    # Add 2x2x2 at (0,0)
    add_block_local(0, 0, 0, 2, 2, 2, U_222, V_222, W_222, 7, U_323, V_323, W_323)
    # Add 2x2x1 at (0,2)
    add_block_local(0, 0, 2, 2, 2, 1, U_221, V_221, W_221, 4, U_323, V_323, W_323)
    # Add 1x2x2 at (2,0)
    add_block_local(2, 0, 0, 1, 2, 2, U_122, V_122, W_122, 4, U_323, V_323, W_323)
    # Add 1x2x1 at (2,2)
    add_block_local(2, 0, 2, 1, 2, 1, U_121, V_121, W_121, 2, U_323, V_323, W_323)
    
    # Assemble 3x2x6
    # B1a (2x3) at (0,0), B1b (2x3) at (0,3)
    U_326 = [[0]*34 for _ in range(6)]
    V_326 = [[0]*34 for _ in range(12)]
    W_326 = [[0]*34 for _ in range(18)]
    
    add_block_local(0, 0, 0, 3, 2, 3, U_323, V_323, W_323, 17, U_326, V_326, W_326)
    add_block_local(0, 0, 3, 3, 2, 3, U_323, V_323, W_323, 17, U_326, V_326, W_326)
    
    # Assemble 3x4x6
    # A1 (3x2) at (0,0), A2 (3x2) at (0,2)
    # B1 (2x6) at (0,0), B2 (2x6) at (2,0)
    
    add_block(0, 0, 0, 3, 2, 6, U_326, V_326, W_326, 34)
    add_block(0, 2, 0, 3, 2, 6, U_326, V_326, W_326, 34)
    
    if time.time() - t0 > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")
        
    return {"U": U, "V": V, "W": W}

# Helper for local assembly
def add_block_local(i_off, j_off, k_off, n_b, m_b, p_b, U_b, V_b, W_b, rank_b, U_out, V_out, W_out):
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                val = U_b[i * m_b + j][r]
                if val != 0:
                    U_out[(i + i_off) * m_b + (j + j_off)][r] = val
        for j in range(m_b):
            for k in range(p_b):
                val = V_b[j * p_b + k][r]
                if val != 0:
                    V_out[(j + j_off) * p_b + (k + k_off)][r] = val
        for k in range(p_b):
            for i in range(n_b):
                val = W_b[k * p_b + i][r]
                if val != 0:
                    W_out[(k + k_off) * n_b + (i + i_off)][r] = val

