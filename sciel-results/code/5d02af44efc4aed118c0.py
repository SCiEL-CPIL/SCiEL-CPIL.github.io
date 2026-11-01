import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    The multiplication is C = A * B, where A is 2x2, B is 2x14, C is 2x14.
    The tensor rank is 49, achieved by applying Strassen's algorithm (rank 7) 
    to each of the 7 disjoint 2x2 blocks of B.
    
    Interface:
    U: (4, 49) coefficients for A
    V: (28, 49) coefficients for B
    W: (28, 49) coefficients for C
    
    Indexing Convention:
    U: row-major flattening of A (2x2) -> indices 0..3
    V: row-major flattening of B (2x14) -> indices 0..27
    W: column-major flattening of C (2x14) -> index = k*2 + i
    """
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # U coefficients (for A): [a, b, c, d]
    # V coefficients (for B): [e, f, g, h]
    # W coefficients (for C): [C00, C01, C10, C11]
    
    # Strassen terms M1..M7
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    A_coeffs = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    B_coeffs = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    
    C_coeffs = [
        [1, 0, 0, 1],    # M1: C00+1, C11+1
        [0, 0, 1, -1],   # M2: C10+1, C11-1
        [0, 1, 0, 1],    # M3: C01+1, C11+1
        [1, 0, 1, 0],    # M4: C00+1, C10+1
        [-1, 1, 0, 0],   # M5: C00-1, C01+1
        [0, 0, 0, 1],    # M6: C11+1
        [1, 0, 0, 0],    # M7: C00+1
    ]
    
    n, m, p = 2, 2, 14
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U (A coefficients)
            # A is 2x2, flattened row-major: [A00, A01, A10, A11]
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
                
            # Fill V (B coefficients)
            # B is 2x14, flattened row-major: [B00...B013, B10...B113]
            # Block b corresponds to columns 2b and 2b+1
            # e = B[0, 2b] -> idx 2b
            # f = B[0, 2b+1] -> idx 2b+1
            # g = B[1, 2b] -> idx 14 + 2b
            # h = B[1, 2b+1] -> idx 14 + 2b + 1
            idx_e = 2 * b
            idx_f = 2 * b + 1
            idx_g = 14 + 2 * b
            idx_h = 15 + 2 * b
            
            V[idx_e][col] = B_coeffs[s][0]
            V[idx_f][col] = B_coeffs[s][1]
            V[idx_g][col] = B_coeffs[s][2]
            V[idx_h][col] = B_coeffs[s][3]
            
            # Fill W (C coefficients)
            # C is 2x14, flattened column-major: [C00, C10, C01, C11, ...]
            # W index = k * n + i = k * 2 + i
            # Block b corresponds to columns 2b and 2b+1
            # C00 (C[0, 2b]) -> idx 2b * 2 + 0 = 4b
            # C10 (C[1, 2b]) -> idx 2b * 2 + 1 = 4b + 1
            # C01 (C[0, 2b+1]) -> idx (2b+1) * 2 + 0 = 4b + 2
            # C11 (C[1, 2b+1]) -> idx (2b+1) * 2 + 1 = 4b + 3
            
            W[4 * b][col] = C_coeffs[s][0]       # C00
            W[4 * b + 1][col] = C_coeffs[s][2]   # C10
            W[4 * b + 2][col] = C_coeffs[s][1]   # C01
            W[4 * b + 3][col] = C_coeffs[s][3]   # C11
            
    return {"U": U, "V": V, "W": W}
