DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # Indices: A (0,0) (0,1) (1,0) (1,1) -> 0,1,2,3
    # B (0,0) (0,1) (1,0) (1,1) -> 0,1,2,3
    # C (0,0) (0,1) (1,0) (1,1) -> 0,1,2,3
    
    # M1 = (a00 + a11)(b00 + b11)
    # U: a00=1, a11=1
    # V: b00=1, b11=1
    # W: c00=1, c11=1
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (i, j, val)
        # V_coeffs: list of (j, k, val)
        # W_coeffs: list of (k, i, val)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]), # M1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1) ]), # M2 = (a10+a11)b00 -> c10
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1) ]), # M3 = a00(b01-b11) -> c01
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (1,0,1) ]), # M4 = a11(b10-b00) -> c10
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1) ]), # M5 = (a00+a01)b11 -> c00 (-1)
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]), # M6 = (a10-a00)(b00+b01) -> c11
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ])  # M7 = (a01-a11)(b10+b11) -> c00
    ]
    
    # Trivial terms for 1x2x2 (Row 2 of A)
    # a20, a21
    # b00, b01, b10, b11
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    trivial_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Note: Local A row is 2. Local B is 0,1. Local C row is 2.
        ([ (2,0,1) ], [ (0,0,1) ], [ (0,2,1) ]), # a20*b00 -> c20
        ([ (2,1,1) ], [ (1,0,1) ], [ (0,2,1) ]), # a21*b10 -> c20
        ([ (2,0,1) ], [ (0,1,1) ], [ (1,2,1) ]), # a20*b01 -> c21
        ([ (2,1,1) ], [ (1,1,1) ], [ (1,2,1) ])  # a21*b11 -> c21
    ]
    
    # Blocks:
    # Block 1: A cols 0,1; B rows 0,1 cols 0,1
    # Block 2: A cols 0,1; B rows 0,1 cols 2,3
    # Block 3: A cols 2,3; B rows 2,3 cols 0,1
    # Block 4: A cols 2,3; B rows 2,3 cols 2,3
    
    blocks = [
        (0, 0), # A col offset, B row/col offset
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    for a_off, b_off in blocks:
        # Add Strassen terms
        for u_c, v_c, w_c in strassen_terms:
            for i, j, val in u_c:
                U[i * m + (a_off + j)][col] += val
            for j, k, val in v_c:
                V[(b_off + j) * p + (b_off + k)][col] += val
            for k, i, val in w_c:
                W[(b_off + k) * n + i][col] += val
            col += 1
            
        # Add Trivial terms
        for u_c, v_c, w_c in trivial_terms:
            for i, j, val in u_c:
                U[i * m + (a_off + j)][col] += val
            for j, k, val in v_c:
                V[(b_off + j) * p + (b_off + k)][col] += val
            for k, i, val in w_c:
                W[(b_off + k) * n + i][col] += val
            col += 1
                
    return {"U": U, "V": V, "W": W}
