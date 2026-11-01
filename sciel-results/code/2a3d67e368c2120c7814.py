"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 35 using Strassen on 5 blocks of 2x2x2
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen terms for 2x2x2
    # U indices: 0: a00, 1: a01, 2: a10, 3: a11
    # V indices for block r: 
    #   b00: 2r, b01: 2r+1, b10: 10+2r, b11: 10+2r+1
    # W indices for block r:
    #   c00: 4r, c01: 4r+1, c10: 4r+2, c11: 4r+3
    # Wait, W index is k*n + i.
    # c00 is C[0,0] for the block? No.
    # C is p x n. C[k, i].
    # For block r, k=2r, 2r+1. i=0, 1.
    # C[2r, 0] -> idx 4r
    # C[2r, 1] -> idx 4r+1
    # C[2r+1, 0] -> idx 4r+2
    # C[2r+1, 1] -> idx 4r+3
    
    # Let's re-verify W indices mapping to c00, c01, c10, c11 in my previous thought.
    # In Strassen, C is 2x2. C[0,0], C[0,1], C[1,0], C[1,1].
    # Here C[0,0] corresponds to k=2r, i=0. Index 4r.
    # C[0,1] corresponds to k=2r, i=1. Index 4r+1.
    # C[1,0] corresponds to k=2r+1, i=0. Index 4r+2.
    # C[1,1] corresponds to k=2r+1, i=1. Index 4r+3.
    
    # My previous mapping:
    # c00 -> C[2r, 0] -> 4r
    # c10 -> C[2r, 1] -> 4r+1
    # c01 -> C[2r+1, 0] -> 4r+2
    # c11 -> C[2r+1, 1] -> 4r+3
    
    # Let's check the Strassen formulas again with this mapping.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So:
    # W[4r] (C[0,0]) gets: M1(+), M4(+), M5(-), M7(+)
    # W[4r+1] (C[0,1]) gets: M3(+), M5(+)
    # W[4r+2] (C[1,0]) gets: M2(+), M4(+)
    # W[4r+3] (C[1,1]) gets: M1(+), M2(-), M3(+), M6(+)
    
    # Let's re-evaluate my previous term assignments.
    # Term 1 (M1): U: a00, a11. V: b00, b11. W: C[0,0](+), C[1,1](+).
    #   W[4r] += 1, W[4r+3] += 1. Correct.
    # Term 2 (M2): U: a10, a11. V: b00. W: C[1,0](+), C[1,1](-).
    #   W[4r+2] += 1, W[4r+3] += -1.
    #   Previous: W[4r+1] (which was C[1,0] in my old naming?)
    #   My old naming: c10 was C[2r, 1] which is C[0,1] in matrix notation?
    #   Matrix C is usually C[i, j] = sum A[i, k] B[k, j].
    #   Here output is C[k, i] = sum A[i, j] B[j, k].
    #   So A is 2x2 (i, j), B is 2x10 (j, k).
    #   Output C is 10x2 (k, i).
    #   For block r, k in {2r, 2r+1}, i in {0, 1}.
    #   C[2r, 0] is element (0,0) of the block result?
    #   Let's call the block result C_r. C_r is 2x2.
    #   C_r[0, 0] = C[2r, 0]. Index 4r.
    #   C_r[0, 1] = C[2r, 1]. Index 4r+1.
    #   C_r[1, 0] = C[2r+1, 0]. Index 4r+2.
    #   C_r[1, 1] = C[2r+1, 1]. Index 4r+3.
    
    # Standard Strassen for C = A * B (2x2 * 2x2):
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So:
    # W[4r] (C_r[0,0]): M1(+), M4(+), M5(-), M7(+)
    # W[4r+1] (C_r[0,1]): M3(+), M5(+)
    # W[4r+2] (C_r[1,0]): M2(+), M4(+)
    # W[4r+3] (C_r[1,1]): M1(+), M2(-), M3(+), M6(+)
    
    # Let's update the term definitions based on this correct mapping.
    
    for r in range(5):
        base = 7 * r
        # V indices
        v_b00 = 2 * r
        v_b01 = 2 * r + 1
        v_b10 = 10 + 2 * r
        v_b11 = 10 + 2 * r + 1
        
        # W indices
        w_c00 = 4 * r
        w_c01 = 4 * r + 1
        w_c10 = 4 * r + 2
        w_c11 = 4 * r + 3
        
        # Term 1: M1 = (a00 + a11)(b00 + b11)
        c = base
        U[0][c] += 1  # a00
        U[3][c] += 1  # a11
        V[v_b00][c] += 1
        V[v_b11][c] += 1
        W[w_c00][c] += 1
        W[w_c11][c] += 1
        
        # Term 2: M2 = (a10 + a11)b00
        c = base + 1
        U[2][c] += 1  # a10
        U[3][c] += 1  # a11
        V[v_b00][c] += 1
        W[w_c10][c] += 1  # C[1,0]
        W[w_c11][c] += -1 # C[1,1]
        
        # Term 3: M3 = a00(b01 - b11)
        c = base + 2
        U[0][c] += 1  # a00
        V[v_b01][c] += 1
        V[v_b11][c] += -1
        W[w_c01][c] += 1  # C[0,1]
        W[w_c11][c] += 1  # C[1,1]
        
        # Term 4: M4 = a11(b10 - b00)
        c = base + 3
        U[3][c] += 1  # a11
        V[v_b10][c] += 1
        V[v_b00][c] += -1
        W[w_c00][c] += 1  # C[0,0]
        W[w_c10][c] += 1  # C[1,0]
        
        # Term 5: M5 = (a00 + a01)b11
        c = base + 4
        U[0][c] += 1  # a00
        U[1][c] += 1  # a01
        V[v_b11][c] += 1
        W[w_c00][c] += -1 # C[0,0]
        W[w_c01][c] += 1  # C[0,1]
        
        # Term 6: M6 = (a10 - a00)(b00 + b01)
        c = base + 5
        U[2][c] += 1  # a10
        U[0][c] += -1 # a00
        V[v_b00][c] += 1
        V[v_b01][c] += 1
        W[w_c11][c] += 1  # C[1,1]
        
        # Term 7: M7 = (a01 - a11)(b10 + b11)
        c = base + 6
        U[1][c] += 1  # a01
        U[3][c] += -1 # a11
        V[v_b10][c] += 1
        V[v_b11][c] += 1
        W[w_c00][c] += 1  # C[0,0]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
