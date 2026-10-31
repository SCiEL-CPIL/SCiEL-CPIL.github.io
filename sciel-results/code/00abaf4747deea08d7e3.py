import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4 matrix multiplication
    # Using two levels of Strassen's algorithm
    
    # 1. Define Strassen coefficients for 2x2 matrices
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # u, v, w vectors for the 7 Strassen steps
    # u, v are length 4 (input A, B)
    # w is length 4 (output C)
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 1, 1, 1]
    w1 = [1, 0, 0, 1] # C11 += M1, C22 += M1
    
    # M2 = (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C21 += M2, C22 -= M2
    
    # M3 = a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # C12 += M3, C22 += M3
    
    # M4 = d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 0, 0] # C11 += M4
    
    # M5 = (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # C11 -= M5, C12 += M5
    
    # M6 = (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 1, 1] # C21 += M6, C22 += M6
    
    # M7 = (b-d)(g+h)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C11 += M7
    
    inner_steps = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # 2. Define outer Strassen steps for 4x4 blocks
    # Blocks A1, A2, A3, A4 correspond to indices 1,2,3,4 in 1-based, or 0,1,2,3 in 0-based
    # A1: (0,0), A2: (0,1), A3: (1,0), A4: (1,1)
    
    # M1 = (A1 + A4)(B1 + B4)
    # X1 = A1 + A4, Y1 = B1 + B4
    # Coeffs for X: A1:1, A2:0, A3:0, A4:1
    # Coeffs for Y: B1:1, B2:0, B3:0, B4:1
    # Contribution to C blocks: C1 += M1, C4 += M1
    outer_step_1 = {
        'x_coeffs': [1, 0, 0, 1], # A1, A2, A3, A4
        'y_coeffs': [1, 0, 0, 1], # B1, B2, B3, B4
        'c_coeffs': [1, 0, 0, 1]  # C1, C2, C3, C4
    }
    
    # M2 = (A3 + A4)B1
    # X2 = A3 + A4, Y2 = B1
    # Coeffs for X: A1:0, A2:0, A3:1, A4:1
    # Coeffs for Y: B1:1, B2:0, B3:0, B4:0
    # Contribution to C blocks: C3 += M2, C4 -= M2
    outer_step_2 = {
        'x_coeffs': [0, 0, 1, 1],
        'y_coeffs': [1, 0, 0, 0],
        'c_coeffs': [0, 0, 1, -1]
    }
    
    # M3 = A1(B2 - B4)
    # X3 = A1, Y3 = B2 - B4
    # Coeffs for X: A1:1, A2:0, A3:0, A4:0
    # Coeffs for Y: B1:0, B2:1, B3:0, B4:-1
    # Contribution to C blocks: C2 += M3, C4 += M3
    outer_step_3 = {
        'x_coeffs': [1, 0, 0, 0],
        'y_coeffs': [0, 1, 0, -1],
        'c_coeffs': [0, 1, 0, 1]
    }
    
    # M4 = A4(B3 - B1)
    # X4 = A4, Y4 = B3 - B1
    # Coeffs for X: A1:0, A2:0, A3:0, A4:1
    # Coeffs for Y: B1:-1, B2:0, B3:1, B4:0
    # Contribution to C blocks: C1 += M4
    outer_step_4 = {
        'x_coeffs': [0, 0, 0, 1],
        'y_coeffs': [-1, 0, 1, 0],
        'c_coeffs': [1, 0, 0, 0]
    }
    
    # M5 = (A1 + A2)B4
    # X5 = A1 + A2, Y5 = B4
    # Coeffs for X: A1:1, A2:1, A3:0, A4:0
    # Coeffs for Y: B1:0, B2:0, B3:0, B4:1
    # Contribution to C blocks: C1 -= M5, C2 += M5
    outer_step_5 = {
        'x_coeffs': [1, 1, 0, 0],
        'y_coeffs': [0, 0, 0, 1],
        'c_coeffs': [-1, 1, 0, 0]
    }
    
    # M6 = (A3 - A1)(B1 + B2)
    # X6 = A3 - A1, Y6 = B1 + B2
    # Coeffs for X: A1:-1, A2:0, A3:1, A4:0
    # Coeffs for Y: B1:1, B2:1, B3:0, B4:0
    # Contribution to C blocks: C3 += M6, C4 += M6
    outer_step_6 = {
        'x_coeffs': [-1, 0, 1, 0],
        'y_coeffs': [1, 1, 0, 0],
        'c_coeffs': [0, 0, 1, 1]
    }
    
    # M7 = (A2 - A4)(B3 + B4)
    # X7 = A2 - A4, Y7 = B3 + B4
    # Coeffs for X: A1:0, A2:1, A3:0, A4:-1
    # Coeffs for Y: B1:0, B2:0, B3:1, B4:1
    # Contribution to C blocks: C1 += M7
    outer_step_7 = {
        'x_coeffs': [0, 1, 0, -1],
        'y_coeffs': [0, 0, 1, 1],
        'c_coeffs': [1, 0, 0, 0]
    }
    
    outer_steps = [
        outer_step_1, outer_step_2, outer_step_3, outer_step_4,
        outer_step_5, outer_step_6, outer_step_7
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(7):
        outer = outer_steps[l]
        x_coeffs = outer['x_coeffs'] # [A1, A2, A3, A4]
        y_coeffs = outer['y_coeffs'] # [B1, B2, B3, B4]
        c_coeffs = outer['c_coeffs'] # [C1, C2, C3, C4]
        
        for s in range(7):
            u_inner, v_inner, w_inner = inner_steps[s]
            
            # Construct U vector for this term
            # u_inner is [u00, u01, u10, u11] for the 2x2 matrix X
            # X = sum_k x_coeffs[k] * A_k
            # X_ab = sum_k x_coeffs[k] * A_k_ab
            # u_inner(X) = sum_ab u_inner[ab] * X_ab = sum_ab u_inner[ab] * sum_k x_coeffs[k] * A_k_ab
            # = sum_k x_coeffs[k] * sum_ab u_inner[ab] * A_k_ab
            
            # A_k_ab corresponds to global index:
            # k=0 (A1): row 0, col 0 -> global (0,0), (0,1), (1,0), (1,1)
            # k=1 (A2): row 0, col 1 -> global (0,2), (0,3), (1,2), (1,3)
            # k=2 (A3): row 1, col 0 -> global (2,0), (2,1), (3,0), (3,1)
            # k=3 (A4): row 1, col 1 -> global (2,2), (2,3), (3,2), (3,3)
            
            for k in range(4):
                if x_coeffs[k] == 0:
                    continue
                # Map k to block row/col
                block_row = k // 2
                block_col = k % 2
                
                for a in range(2):
                    for b in range(2):
                        # u_inner index: a*2 + b? No, u_inner is [00, 01, 10, 11]
                        # So index is a*2 + b
                        u_val = u_inner[a * 2 + b]
                        if u_val == 0:
                            continue
                        coeff = x_coeffs[k] * u_val
                        # Global index in A: (block_row*2 + a, block_col*2 + b)
                        gi = block_row * 2 + a
                        gj = block_col * 2 + b
                        U[gi * m + gj][col] += coeff
                        
            # Construct V vector for this term
            for k in range(4):
                if y_coeffs[k] == 0:
                    continue
                block_row = k // 2
                block_col = k % 2
                
                for a in range(2):
                    for b in range(2):
                        v_val = v_inner[a * 2 + b]
                        if v_val == 0:
                            continue
                        coeff = y_coeffs[k] * v_val
                        # Global index in B: (block_row*2 + a, block_col*2 + b)
                        gi = block_row * 2 + a
                        gj = block_col * 2 + b
                        V[gi * p + gj][col] += coeff
                        
            # Construct W vector for this term
            # w_inner is [w00, w01, w10, w11] for the 2x2 result C_l
            # C_l_ab = sum_s w_inner[ab] * (u_s(X) v_s(Y))
            # Global C block c gets contribution c_coeffs[c] * C_l_ab
            # Global C index: (c_row*2 + a, c_col*2 + b)
            # W index convention: W[k*n + i, r] where k is col, i is row?
            # Task says: W[k*n+i, r].
            # Standard matrix C[i,k].
            # Here output is C[i,k].
            # Block c has row c_row, col c_col.
            # Global row i = c_row*2 + a
            # Global col k = c_col*2 + b
            # W index = k*n + i = (c_col*2 + b)*n + (c_row*2 + a)
            
            for c in range(4):
                if c_coeffs[c] == 0:
                    continue
                c_row = c // 2
                c_col = c % 2
                
                for a in range(2):
                    for b in range(2):
                        w_val = w_inner[a * 2 + b]
                        if w_val == 0:
                            continue
                        coeff = c_coeffs[c] * w_val
                        gi = c_row * 2 + a
                        gj = c_col * 2 + b
                        # W index: gj * n + gi
                        W[gj * n + gi][col] += coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
