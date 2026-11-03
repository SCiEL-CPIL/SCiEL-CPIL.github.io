"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split 5 into 3 and 2
    n1, n2 = 3, 2
    m1, m2 = 3, 2
    p1, p2 = 3, 2
    
    # Total rank
    rank = 27 + 18 + 18 + 6 + 18 + 6 + 4 + 7
    # 27: 3x3x3
    # 18: 3x2x3
    # 18: 3x3x2
    # 6: 3x2x2
    # 18: 2x3x3
    # 6: 2x2x3
    # 4: 2x3x2
    # 7: 2x2x2 (Strassen)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        nonlocal col
        num_cols = len(U_block[0])
        for r in range(len(U_block)):
            for c in range(num_cols):
                if U_block[r][c] != 0:
                    U[u_offset + r][col + c] += U_block[r][c]
        for r in range(len(V_block)):
            for c in range(num_cols):
                if V_block[r][c] != 0:
                    V[v_offset + r][col + c] += V_block[r][c]
        for r in range(len(W_block)):
            for c in range(num_cols):
                if W_block[r][c] != 0:
                    W[w_offset + r][col + c] += W_block[r][c]
        col += num_cols

    # 1. A33 B33 (3x3x3, Schoolbook Rank 27)
    U_b = [[0]*27 for _ in range(9)]
    V_b = [[0]*27 for _ in range(9)]
    W_b = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i*3 + j
                U_b[idx][col] = 1
                V_b[j*3 + k][col] = 1
                W_b[k*3 + i][col] = 1
                col += 1
    add_block(U_b, V_b, W_b, 0, 0, 0) # A[0:3, 0:3], B[0:3, 0:3], C[0:3, 0:3]
    # Note: add_block uses global col, but I incremented col inside loop.
    # I should reset col or manage it carefully.
    # Let's restart col management.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. A33 B33
    U_b = [[0]*27 for _ in range(9)]
    V_b = [[0]*27 for _ in range(9)]
    W_b = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i*3 + j
                U_b[idx][col] = 1
                V_b[j*3 + k][col] = 1
                W_b[k*3 + i][col] = 1
                col += 1
    add_block(U_b, V_b, W_b, 0, 0, 0)

    # 2. A32 B23 (3x2x3, Rank 18)
    # A is 3x2, B is 2x3, C is 3x3
    U_b = [[0]*18 for _ in range(6)]
    V_b = [[0]*18 for _ in range(6)]
    W_b = [[0]*18 for _ in range(9)]
    for i in range(3):
        for j in range(2):
            for k in range(3):
                idx = i*2 + j
                U_b[idx][col] = 1
                V_b[j*3 + k][col] = 1
                W_b[k*3 + i][col] = 1
                col += 1
    # A is at rows 0-2, cols 3-4 (offset 3)
    # B is at rows 3-4, cols 0-2 (offset 3)
    # C is at rows 0-2, cols 0-2 (offset 0)
    add_block(U_b, V_b, W_b, 3, 3, 0)

    # 3. A33 B32 (3x3x2, Rank 18)
    # A is 3x3, B is 3x2, C is 3x2
    U_b = [[0]*18 for _ in range(9)]
    V_b = [[0]*18 for _ in range(6)]
    W_b = [[0]*18 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            for k in range(2):
                idx = i*3 + j
                U_b[idx][col] = 1
                V_b[j*2 + k][col] = 1
                W_b[k*3 + i][col] = 1
                col += 1
    # A is at 0, B is at 3 (cols 3-4), C is at 3 (cols 3-4)
    add_block(U_b, V_b, W_b, 0, 3, 3)

    # 4. A32 B22 (3x2x2, Rank 6)
    # A is 3x2, B is 2x2, C is 3x2
    U_b = [[0]*6 for _ in range(6)]
    V_b = [[0]*6 for _ in range(4)]
    W_b = [[0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(2):
            for k in range(2):
                idx = i*2 + j
                U_b[idx][col] = 1
                V_b[j*2 + k][col] = 1
                W_b[k*3 + i][col] = 1
                col += 1
    # A is at 3, B is at 6 (rows 3-4, cols 3-4), C is at 3
    add_block(U_b, V_b, W_b, 3, 6, 3)

    # 5. A23 B33 (2x3x3, Rank 18)
    # A is 2x3, B is 3x3, C is 2x3
    U_b = [[0]*18 for _ in range(6)]
    V_b = [[0]*18 for _ in range(9)]
    W_b = [[0]*18 for _ in range(6)]
    for i in range(2):
        for j in range(3):
            for k in range(3):
                idx = i*3 + j
                U_b[idx][col] = 1
                V_b[j*3 + k][col] = 1
                W_b[k*2 + i][col] = 1
                col += 1
    # A is at 6 (rows 3-4, cols 0-2), B is at 0, C is at 6 (rows 3-4, cols 0-2)
    add_block(U_b, V_b, W_b, 6, 0, 6)

    # 6. A22 B23 (2x2x3, Rank 6)
    # A is 2x2, B is 2x3, C is 2x3
    U_b = [[0]*6 for _ in range(4)]
    V_b = [[0]*6 for _ in range(6)]
    W_b = [[0]*6 for _ in range(6)]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                idx = i*2 + j
                U_b[idx][col] = 1
                V_b[j*3 + k][col] = 1
                W_b[k*2 + i][col] = 1
                col += 1
    # A is at 6, B is at 3, C is at 6
    add_block(U_b, V_b, W_b, 6, 3, 6)

    # 7. A23 B32 (2x3x2, Rank 4)
    # A is 2x3, B is 3x2, C is 2x2
    U_b = [[0]*4 for _ in range(6)]
    V_b = [[0]*4 for _ in range(6)]
    W_b = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                idx = i*3 + j
                U_b[idx][col] = 1
                V_b[j*2 + k][col] = 1
                W_b[k*2 + i][col] = 1
                col += 1
    # A is at 6, B is at 3, C is at 6
    add_block(U_b, V_b, W_b, 6, 3, 6)

    # 8. A22 B22 (2x2x2, Strassen Rank 7)
    # Strassen U, V, W
    U_b = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]
    V_b = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    W_b = [
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    # Note: Strassen indices need to be checked.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # My U_b, V_b, W_b above are placeholders. I need correct Strassen.
    # Let's use the standard one.
    U_b = [
        [1, 1, 0, 0], # A11+A22
        [0, 0, 1, 1], # A21+A22
        [1, 0, 0, 0], # A11
        [0, 0, 0, 1], # A22
        [1, 1, 0, 0], # A11+A12
        [1, -1, 0, 0], # A21-A11
        [0, 0, 1, -1] # A12-A22
    ]
    V_b = [
        [1, 1, 0, 0], # B11+B22
        [1, 0, 0, 0], # B11
        [0, 0, 1, -1], # B21-B22
        [0, 1, -1, 0], # B12-B11
        [0, 0, 0, 1], # B22
        [1, 1, 0, 0], # B11+B12
        [0, 0, 1, 1] # B21+B22
    ]
    W_b = [
        [1, 0, 0, 0], # C11
        [0, 0, 1, 0], # C21
        [0, 1, 0, 0], # C12
        [0, 0, 0, 1], # C22
        [0, 1, 0, 0], # C12
        [0, 0, 1, 0], # C21
        [1, 0, 0, 0]  # C11
    ]
    # Wait, C11 = M1+M4-M5+M7.
    # So W for M1 has 1 at C11.
    # W for M4 has 1 at C11.
    # W for M5 has -1 at C11.
    # W for M7 has 1 at C11.
    
    # Let's re-verify Strassen W coefficients.
    # C11: M1, M4, -M5, M7
    # C12: M3, M5
    # C21: M2, M4
    # C22: M1, -M2, M3, M6
    
    W_b = [
        [1, 0, 0, 0], # M1 -> C11
        [0, 0, 1, 0], # M2 -> C21
        [0, 1, 0, 0], # M3 -> C12
        [1, 0, 0, 0], # M4 -> C11
        [-1, 0, 1, 0], # M5 -> -C11, C12
        [0, 0, 0, 1], # M6 -> C22
        [1, 0, 0, 0]  # M7 -> C11
    ]
    # M5: (A11+A12)B22. C12 += M5. C11 -= M5.
    # So W5 should have -1 at C11 and 1 at C12.
    # My W_b[4] is [-1, 0, 1, 0]. Correct.
    # M6: (A21-A11)(B11+B12). C22 += M6.
    # My W_b[5] is [0, 0, 0, 1]. Correct.
    # M7: (A12-A22)(B21+B22). C11 += M7.
    # My W_b[6] is [1, 0, 0, 0]. Correct.
    
    add_block(U_b, V_b, W_b, 6, 6, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
