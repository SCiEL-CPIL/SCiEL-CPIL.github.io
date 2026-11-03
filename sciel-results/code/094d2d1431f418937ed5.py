import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # a_r, b_r, c_r are flattened 2x2 matrices (length 4)
    # Order: [00, 01, 10, 11]
    a_coeffs = [
        [1, 0, 0, 1],  # 1
        [0, 0, 1, 1],  # 2
        [1, 0, 0, 0],  # 3
        [0, 0, 0, 1],  # 4
        [1, 1, 0, 0],  # 5
        [-1, 0, 1, 0], # 6
        [0, 1, 0, -1]  # 7
    ]
    b_coeffs = [
        [1, 0, 0, 1],  # 1
        [1, 0, 0, 0],  # 2
        [0, 1, 0, -1], # 3
        [0, 0, 1, -1], # 4
        [0, 0, 0, 1],  # 5
        [1, 1, 0, 0],  # 6
        [0, 0, 1, 1]   # 7
    ]
    c_coeffs = [
        [1, 0, 0, 1],  # 1: C00, C11
        [0, 0, 1, -1], # 2: C10, C11
        [0, 1, 0, 1],  # 3: C01, C11
        [1, 0, 1, 0],  # 4: C00, C10
        [-1, 1, 0, 0], # 5: C00, C01
        [0, 0, 0, 1],  # 6: C11
        [1, 0, 0, 0]   # 7: C00
    ]
    
    # Block combination coefficients for 4x4x4
    # alpha_k[i,j], beta_k[l,m], gamma_k[c]
    # Blocks indexed 0..3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # 0:00, 1:01, 2:10, 3:11
    alpha_coeffs = [
        [1, 0, 0, 1],  # 1: A00+A11
        [0, 0, 1, 1],  # 2: A10+A11
        [1, 0, 0, 0],  # 3: A00
        [0, 0, 0, 1],  # 4: A11
        [1, 1, 0, 0],  # 5: A00+A01
        [-1, 0, 1, 0], # 6: A10-A00
        [0, 1, 0, -1]  # 7: A01-A11
    ]
    beta_coeffs = [
        [1, 0, 0, 1],  # 1: B00+B11
        [1, 0, 0, 0],  # 2: B00
        [0, 1, 0, -1], # 3: B01-B11
        [0, 0, 1, -1], # 4: B10-B00
        [0, 0, 0, 1],  # 5: B11
        [1, 1, 0, 0],  # 6: B00+B01
        [0, 0, 1, 1]   # 7: B10+B11
    ]
    gamma_coeffs = [
        [1, 0, 0, 1],  # 1: C00, C11
        [0, 0, 1, -1], # 2: C10, C11
        [0, 1, 0, 1],  # 3: C01, C11
        [1, 0, 1, 0],  # 4: C00, C10
        [-1, 1, 0, 0], # 5: C00, C01
        [0, 0, 0, 1],  # 6: C11
        [1, 0, 0, 0]   # 7: C00
    ]
    
    rank = 14 * 7 # 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Two halves for p=8
    for half in range(2):
        for k in range(7):
            for r in range(7):
                # U: 16 entries
                # A blocks 0..3. Entry (u,v) in block (I,J) is at 4*(2*I+u) + (2*J+v)
                for ij in range(4):
                    I = ij // 2
                    J = ij % 2
                    alpha_val = alpha_coeffs[k][ij]
                    if alpha_val != 0:
                        for u in range(2):
                            for v in range(2):
                                a_val = a_coeffs[r][u*2+v]
                                if a_val != 0:
                                    flat_idx = 4 * (2*I + u) + (2*J + v)
                                    U[flat_idx][col] += alpha_val * a_val
                                    
                # V: 32 entries
                # B blocks 0..3. Entry (u,v) in block (I,J) is at 8*(2*I+u) + (2*J+v)
                # Half offset: if half=1, col offset is 4
                for lm in range(4):
                    I = lm // 2
                    J = lm % 2
                    beta_val = beta_coeffs[k][lm]
                    if beta_val != 0:
                        for u in range(2):
                            for v in range(2):
                                b_val = b_coeffs[r][u*2+v]
                                if b_val != 0:
                                    flat_idx = 8 * (2*I + u) + (2*J + v + 4*half)
                                    V[flat_idx][col] += beta_val * b_val
                                    
                # W: 32 entries
                # C blocks 0..3. Entry (u,v) in block (I,J) is at 4*(2*J+v) + (2*I+u) ?
                # W shape (p*n, rank) = 8*4.
                # W[k*n+i, r]. k is row of C (0..7), i is col of C (0..3).
                # k = 2*J + v, i = 2*I + u.
                # flat_idx = k*4 + i.
                for c_block in range(4):
                    I = c_block // 2
                    J = c_block % 2
                    gamma_val = gamma_coeffs[k][c_block]
                    if gamma_val != 0:
                        for u in range(2):
                            for v in range(2):
                                c_val = c_coeffs[r][u*2+v]
                                if c_val != 0:
                                    i = 2*I + u
                                    k_row = 2*J + v + 4*half
                                    flat_idx = k_row * 4 + i
                                    W[flat_idx][col] += gamma_val * c_val
                                    
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
