import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49 using Strassen for 7 blocks of 2x2
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # U: 2x2 indices (i, j) -> flat i*m + j
    # V: 2x2 indices (j, k) -> flat j*2 + k (local)
    # W: 2x2 indices (k, i) -> flat k*n + i (local)
    
    # Define the 7 terms for one 2x2x2 block
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): val}
    # V_coeffs: dict {(j,k): val}
    # W_coeffs: dict {(k,i): val}
    
    terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # 7 blocks
    for b in range(7):
        for t_idx in range(7):
            col = b * 7 + t_idx
            u_dict, v_dict, w_dict = terms[t_idx]
            
            for (i, j), val in u_dict.items():
                U[i * m + j][col] += val
                
            for (j, k), val in v_dict.items():
                # Global V index: j * p + (b * 2 + k)
                V[j * p + b * 2 + k][col] += val
                
            for (k, i), val in w_dict.items():
                # Global W index: (b * 2 + k) * n + i
                W[(b * 2 + k) * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
