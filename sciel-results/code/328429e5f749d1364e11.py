import time

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Returns the rank-21 Strassen-based decomposition for M_{2,2,6}.
    
    The matrix multiplication tensor for (2x2) @ (2x6) is decomposed into
    three independent 2x2 block multiplications, each using Strassen's
    7-product algorithm.
    
    U: shape (n*m, rank) = (4, 21)
    V: shape (m*p, rank) = (12, 21)
    W: shape (p*n, rank) = (12, 21)
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 products
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B local indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: (row i, col k_local)
    
    # U coefficients: which A entries participate in each product
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients: list of (local_row, local_col, val) for each product
    # local_row: 0 or 1 (row of B block), local_col: 0 or 1 (col within block)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # W coefficients: list of (row_i, local_col, val) for each product
    # row_i: 0 or 1 (row of C), local_col: 0 or 1 (col within block)
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: C00 += M1, C11 += M1
        [(1, 0, 1), (1, 1, -1)],  # M2: C10 += M2, C11 -= M2
        [(0, 1, 1), (1, 1, 1)],   # M3: C01 += M3, C11 += M3
        [(0, 0, 1), (1, 0, 1)],   # M4: C00 += M4, C10 += M4
        [(0, 0, -1), (0, 1, 1)],  # M5: C00 -= M5, C01 += M5
        [(1, 1, 1)],              # M6: C11 += M6
        [(0, 0, 1)]               # M7: C00 += M7
    ]
    
    for b in range(3):  # Three 2x2 blocks
        for t in range(7):
            r = b * 7 + t
            
            # Fill U entries
            for idx, val in U_coeffs[t].items():
                U[idx][r] = val
            
            # Fill V entries
            # Global V index: row * p + col, where col = 2*b + local_col
            for v_row, v_col_local, val in V_coeffs[t]:
                global_col = 2 * b + v_col_local
                v_idx = v_row * p + global_col
                V[v_idx][r] = val
            
            # Fill W entries
            # Global W index: col * n + row, where col = 2*b + local_col
            for w_row, w_col_local, val in W_coeffs[t]:
                global_col = 2 * b + w_col_local
                w_idx = global_col * n + w_row
                W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
