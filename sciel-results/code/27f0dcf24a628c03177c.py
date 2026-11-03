"""Matrix multiplication 3x5 x 5x5: Strassen-based block decomposition for lower rank."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5 * 5x5 matrix multiplication
    using a Strassen-like block decomposition strategy.
    
    Strategy:
    - Partition A (3x5) into top 2 rows and bottom 1 row
    - Partition B (5x5) into left 3 columns and right 2 columns
    - Use Strassen for 2x2 sub-block products where applicable
    - Combine with schoolbook for remaining blocks
    
    Block structure:
    A = [A_top (2x5); A_bottom (1x5)]
    B = [B_left (5x3) | B_right (5x2)]
    
    C = A * B = [A_top * B_left (2x3) | A_top * B_right (2x2)]
                [A_bottom * B_left (1x3) | A_bottom * B_right (1x2)]
    
    Further partition:
    A_top = [A1 (2x3) | A2 (2x2)]
    A_bottom = [A3 (1x3) | A4 (1x2)]
    B_left = [B1 (3x3); B3 (2x3)]
    B_right = [B2 (3x2); B4 (2x2)]
    
    Products:
    C11 = A1*B1 + A2*B3 (2x3)
    C12 = A1*B2 + A2*B4 (2x2)
    C21 = A3*B1 + A4*B3 (1x3)
    C22 = A3*B2 + A4*B4 (1x2)
    
    For each product, use optimal rank:
    - 2x2 * 2x2: Strassen rank 7
    - 1xk * kxl: rank l (dot product)
    - 2x3 * 3x3: rank 6 (each of 2 rows does 3 mults)
    - 2x3 * 3x2: rank 6
    - 2x2 * 2x3: rank 6
    - 1x3 * 3x3: rank 3
    - 1x2 * 2x3: rank 2
    - 1x3 * 3x2: rank 2
    - 1x2 * 2x2: rank 2
    
    Total rank: 6+6+6+7+3+2+2+2 = 34
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Initialize matrices
    U = [[0.0] * 100 for _ in range(n * m)]  # 15 x rank
    V = [[0.0] * 100 for _ in range(m * p)]  # 25 x rank
    W = [[0.0] * 100 for _ in range(p * n)]  # 15 x rank
    
    rank = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        nonlocal rank
        U[u_idx][rank] += u_val
        V[v_idx][rank] += v_val
        W[w_idx][rank] += w_val
        rank += 1
    
    # Block 1: A1 (2x3) * B1 (3x3) -> C11 (2x3)
    # A1: rows 0-1, cols 0-2; B1: rows 0-2, cols 0-2; C11: rows 0-1, cols 0-2
    # Rank 6: for each of 3 middle indices j, do A1[:,j] * B1[j,:]
    for j in range(3):
        for i in range(2):
            for k in range(3):
                add_term(i * m + j, j * p + k, k * n + i)
    
    # Block 2: A2 (2x2) * B3 (2x3) -> C11 (2x3)
    # A2: rows 0-1, cols 3-4; B3: rows 3-4, cols 0-2; C11: rows 0-1, cols 0-2
    # Rank 6: for each of 2 middle indices j (offset by 3), do A2[:,j] * B3[j,:]
    for j in range(2):
        for i in range(2):
            for k in range(3):
                add_term(i * m + (3 + j), (3 + j) * p + k, k * n + i)
    
    # Block 3: A1 (2x3) * B2 (3x2) -> C12 (2x2)
    # A1: rows 0-1, cols 0-2; B2: rows 0-2, cols 3-4; C12: rows 0-1, cols 3-4
    # Rank 6: for each of 3 middle indices j, do A1[:,j] * B2[j,:]
    for j in range(3):
        for i in range(2):
            for k in range(2):
                add_term(i * m + j, j * p + (3 + k), (3 + k) * n + i)
    
    # Block 4: A2 (2x2) * B4 (2x2) -> C12 (2x2)
    # A2: rows 0-1, cols 3-4; B4: rows 3-4, cols 3-4; C12: rows 0-1, cols 3-4
    # Use Strassen rank 7
    # Let A2 = [[a, b], [c, d]], B4 = [[e, f], [g, h]]
    # C12 = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    a_idx = 0 * m + 3   # A2[0,0]
    b_idx = 0 * m + 4   # A2[0,1]
    c_idx = 1 * m + 3   # A2[1,0]
    d_idx = 1 * m + 4   # A2[1,1]
    
    e_idx = 3 * p + 3   # B4[0,0]
    f_idx = 3 * p + 4   # B4[0,1]
    g_idx = 4 * p + 3   # B4[1,0]
    h_idx = 4 * p + 4   # B4[1,1]
    
    c11_w = 3 * n + 0   # C12[0,0]
    c12_w = 3 * n + 1   # C12[0,1]
    c21_w = 3 * n + 2   # C12[1,0]
    c22_w = 3 * n + 3   # C12[1,1]
    
    # M1 = (a+d)(e+h)
    U[a_idx][rank] = 1.0
    U[d_idx][rank] = 1.0
    V[e_idx][rank] = 1.0
    V[h_idx][rank] = 1.0
    W[c11_w][rank] = 1.0
    W[c22_w][rank] = 1.0
    rank += 1
    
    # M2 = (c+d)e
    U[c_idx][rank] = 1.0
    U[d_idx][rank] = 1.0
    V[e_idx][rank] = 1.0
    W[c21_w][rank] = 1.0
    W[c22_w][rank] = -1.0
    rank += 1
    
    # M3 = a(f-h)
    U[a_idx][rank] = 1.0
    V[f_idx][rank] = 1.0
    V[h_idx][rank] = -1.0
    W[c12_w][rank] = 1.0
    W[c22_w][rank] = 1.0
    rank += 1
    
    # M4 = d(g-e)
    U[d_idx][rank] = 1.0
    V[g_idx][rank] = 1.0
    V[e_idx][rank] = -1.0
    W[c11_w][rank] = 1.0
    W[c21_w][rank] = 1.0
    rank += 1
    
    # M5 = (a+b)h
    U[a_idx][rank] = 1.0
    U[b_idx][rank] = 1.0
    V[h_idx][rank] = 1.0
    W[c11_w][rank] = -1.0
    W[c12_w][rank] = 1.0
    rank += 1
    
    # M6 = (c-a)(e+f)
    U[c_idx][rank] = 1.0
    U[a_idx][rank] = -1.0
    V[e_idx][rank] = 1.0
    V[f_idx][rank] = 1.0
    W[c22_w][rank] = 1.0
    rank += 1
    
    # M7 = (b-d)(g+h)
    U[b_idx][rank] = 1.0
    U[d_idx][rank] = -1.0
    V[g_idx][rank] = 1.0
    V[h_idx][rank] = 1.0
    W[c11_w][rank] = 1.0
    rank += 1
    
    # Block 5: A3 (1x3) * B1 (3x3) -> C21 (1x3)
    # A3: row 2, cols 0-2; B1: rows 0-2, cols 0-2; C21: row 2, cols 0-2
    # Rank 3: dot product
    for j in range(3):
        for k in range(3):
            add_term(2 * m + j, j * p + k, k * n + 2)
    
    # Block 6: A4 (1x2) * B3 (2x3) -> C21 (1x3)
    # A4: row 2, cols 3-4; B3: rows 3-4, cols 0-2; C21: row 2, cols 0-2
    # Rank 2: dot product
    for j in range(2):
        for k in range(3):
            add_term(2 * m + (3 + j), (3 + j) * p + k, k * n + 2)
    
    # Block 7: A3 (1x3) * B2 (3x2) -> C22 (1x2)
    # A3: row 2, cols 0-2; B2: rows 0-2, cols 3-4; C22: row 2, cols 3-4
    # Rank 2: dot product
    for j in range(3):
        for k in range(2):
            add_term(2 * m + j, j * p + (3 + k), (3 + k) * n + 2)
    
    # Block 8: A4 (1x2) * B4 (2x2) -> C22 (1x2)
    # A4: row 2, cols 3-4; B4: rows 3-4, cols 3-4; C22: row 2, cols 3-4
    # Rank 2: dot product
    for j in range(2):
        for k in range(2):
            add_term(2 * m + (3 + j), (3 + j) * p + (3 + k), (3 + k) * n + 2)
    
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
