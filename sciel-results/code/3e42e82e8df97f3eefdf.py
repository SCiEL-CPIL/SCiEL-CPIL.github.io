import numpy as np

DIMENSIONS = [4, 4, 8]

def _base_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U rows: A00, A01, A10, A11
    V rows: B00, B01, B10, B11
    W rows: C00, C01, C10, C11
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Product 1: (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += P1, C11 += P1
    
    # Product 2: (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 += P2, C11 -= P2
    
    # Product 3: A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += P3, C11 += P3
    
    # Product 4: A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += P4, C10 += P4
    
    # Product 5: (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -= P5, C01 += P5
    
    # Product 6: (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += P6
    
    # Product 7: (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 += P7
    
    return U, V, W

def _construct_444():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    U: 16x49, V: 16x49, W: 16x49
    """
    U_base, V_base, W_base = _base_strassen()
    
    # Coefficients for A blocks in each of the 7 products
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3)
    # Product 1: A00 + A11
    a_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    # Coefficients for B blocks in each of the 7 products
    # B blocks: B00, B01, B10, B11 (indices 0,1,2,3)
    # Product 1: B00 + B11
    b_coeffs = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # Coefficients for C blocks in each of the 7 products
    # C blocks: C00, C01, C10, C11 (indices 0,1,2,3)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    c_coeffs = [
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, -1],# P2 -> C10, C11
        [0, 1, 0, 1], # P3 -> C01, C11
        [1, 0, 1, 0], # P4 -> C00, C10
        [-1, 1, 0, 0],# P5 -> C00, C01
        [0, 0, 0, 1], # P6 -> C11
        [1, 0, 0, 0]  # P7 -> C00
    ]
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for k in range(7):
        for p in range(7):
            col = k * 7 + p
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    a_block = i // 2
                    b_block = j // 2
                    r = i % 2
                    c = j % 2
                    coeff = a_coeffs[k][a_block * 2 + b_block]
                    if coeff != 0:
                        U[i * 4 + j, col] = coeff * U_base[r * 2 + c, p]
                        
            # Fill V
            for j in range(4):
                for l in range(4):
                    c_block = j // 2
                    d_block = l // 2
                    s = j % 2
                    t = l % 2
                    coeff = b_coeffs[k][c_block * 2 + d_block]
                    if coeff != 0:
                        V[j * 4 + l, col] = coeff * V_base[s * 2 + t, p]
                        
            # Fill W
            for l in range(4):
                for i in range(4):
                    a_block = l // 2
                    b_block = i // 2
                    r = l % 2
                    c = i % 2
                    coeff = c_coeffs[k][a_block * 2 + b_block]
                    if coeff != 0:
                        W[l * 4 + i, col] = coeff * W_base[r * 2 + c, p]
                        
    return U, V, W

def construct():
    """
    Constructs the decomposition for 4x4x8 matrix multiplication.
    Rank: 98
    """
    U_444, V_444, W_444 = _construct_444()
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U is the same for both halves
    U[:, :49] = U_444
    U[:, 49:] = U_444
    
    # V: Left half (cols 0-3 of B) and Right half (cols 4-7 of B)
    for j in range(4):
        for k in range(4):
            # Left half: global k = k
            V[j * 8 + k, :49] = V_444[j * 4 + k, :]
            # Right half: global k = k + 4
            V[j * 8 + (k + 4), 49:] = V_444[j * 4 + k, :]
            
    # W: Left half (cols 0-3 of C) and Right half (cols 4-7 of C)
    # W_444 index: l * 4 + i (l=row, i=col)
    # W_total index: k * 4 + i (k=col, i=row)
    for l in range(4):
        for i in range(4):
            # Left half: global k = i (col in 4x4), row = l
            # Wait, in 4x4x4, C[l, i]. l is row, i is col.
            # In 4x4x8, C[row, col].
            # Left half cols are 0-3.
            # So for C[l, k_local], global index is k_local * 4 + l.
            # W_444 index is l * 4 + k_local.
            W[i * 4 + l, :49] = W_444[l * 4 + i, :] # Wait, mapping check
            
            # Let's re-verify the mapping.
            # W_444[row_idx] where row_idx = l * 4 + i.
            # This corresponds to C[l, i].
            # In total W, we want to place this at row_idx_total = k * 4 + row.
            # Here k = i (col in left half), row = l.
            # So W_total[i * 4 + l] = W_444[l * 4 + i].
            
            # Right half: global k = i + 4.
            W[(i + 4) * 4 + l, 49:] = W_444[l * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
