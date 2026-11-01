import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm for 3 blocks
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U part)
    # Indexing: 0: a00, 1: a01, 2: a10, 3: a11
    U_coeffs = [
        [1, 0, 0, 1],  # r=0: a00 + a11
        [0, 0, 1, 1],  # r=1: a10 + a11
        [1, 0, 0, 0],  # r=2: a00
        [0, 0, 0, 1],  # r=3: a11
        [1, 1, 0, 0],  # r=4: a00 + a01
        [-1, 0, 1, 0], # r=5: a10 - a00
        [0, 1, 0, -1]  # r=6: a01 - a11
    ]
    
    # Strassen coefficients for B (V part)
    # Indexing: 0: b00, 1: b01, 2: b10, 3: b11
    V_coeffs = [
        [1, 0, 0, 1],  # r=0: b00 + b11
        [1, 0, 0, 0],  # r=1: b00
        [0, 1, 0, -1], # r=2: b01 - b11
        [-1, 0, 1, 0], # r=3: b10 - b00
        [0, 0, 0, 1],  # r=4: b11
        [1, 1, 0, 0],  # r=5: b00 + b01
        [0, 0, 1, 1]   # r=6: b10 + b11
    ]
    
    # Strassen reconstruction coefficients for C (W part)
    # Indexing: 0: c00, 1: c01, 2: c10, 3: c11
    W_coeffs = [
        [1, 0, 0, 1],  # r=0
        [0, 0, 1, -1], # r=1
        [0, 1, 0, 1],  # r=2
        [1, 0, 1, 0],  # r=3
        [-1, 1, 0, 0], # r=4
        [0, 0, 0, 1],  # r=5
        [1, 0, 0, 0]   # r=6
    ]
    
    # Fill the matrices
    for b in range(3): # 3 blocks of 2 columns
        for r in range(7): # 7 Strassen products
            col = 7 * b + r
            
            # Fill U
            # U row index: i * m + j = 2 * i + j
            for i in range(2):
                for j in range(2):
                    idx = i * m + j
                    U[idx][col] = U_coeffs[r][idx]
            
            # Fill V
            # V row index: j * p + k = 2 * j + k
            # k is 2*b + l where l is 0 or 1
            for j in range(2):
                for l in range(2):
                    k = 2 * b + l
                    idx = j * p + k
                    # V_coeffs[r] index: j * 2 + l
                    U_val_idx = j * 2 + l
                    V[idx][col] = V_coeffs[r][U_val_idx]
            
            # Fill W
            # W row index: k * n + i = 6 * k + i
            # k is 2*b + l where l is 0 or 1
            for l in range(2):
                k = 2 * b + l
                for i in range(2):
                    idx = k * n + i
                    # W_coeffs[r] index: l * 2 + i
                    W_val_idx = l * 2 + i
                    W[idx][col] = W_coeffs[r][W_val_idx]
                    
    return {"U": U, "V": V, "W": W}
