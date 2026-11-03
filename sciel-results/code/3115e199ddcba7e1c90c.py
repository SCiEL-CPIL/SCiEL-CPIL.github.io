import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split m=4 into 2 blocks of 2.
    # For each block (2 of them):
    #   Split n=3 into 2 (top) and 1 (bot).
    #   Top: 2x2x8. Split p=8 into 4 blocks of 2.
    #         Each 2x2x2 uses Strassen (rank 7).
    #         Total Top Rank = 4 * 7 = 28.
    #   Bot: 1x2x8. Schoolbook rank = 1*2*8 = 16.
    #   Total per m-block = 28 + 16 = 44.
    # Total Rank = 2 * 44 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, -C11
    # M3 = A00(B01-B11) -> C01, C11
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> -C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    strassen_decomp = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for j1 in range(2):
        # Top part: 2x2x8
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # A (U)
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                
                # B (V)
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                
                # C (W)
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8
        # Rank 16
        for k in range(8):
            for j2 in range(2):
                # Term: A[2, 2*j1+j2] * B[2*j1+j2, k] * C[2, k]
                
                # U
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                
                # V
                global_j = 2 * j1 + j2
                global_k = k
                V[global_j * p + global_k][col] = 1
                
                # W
                global_i = 2
                global_k = k
                W[global_k * n + global_i][col] = 1
                
                col += 1

    if time.time() > deadline:
        # Fallback: return current best (which is the constructed one)
        pass

    return {"U": U, "V": V, "W": W}
