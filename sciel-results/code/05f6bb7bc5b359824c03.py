import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Define Strassen coefficients for 2x2x2
    # Each term is a tuple (u_indices, v_indices, w_indices)
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k, coeff)
    # w_indices: list of (i, k, coeff)
    
    # We will process 8 blocks.
    # Block b is defined by j_offset (0 or 2) and k_offset (0, 2, 4, 6)
    
    col_idx = 0
    
    # Strassen 7 multiplications
    # P1: (a+d)(e+h)
    # P2: (c+d)e
    # P3: a(f-h)
    # P4: d(g-e)
    # P5: (a+b)h
    # P6: (c-a)(e+f)
    # P7: (b-d)(g+h)
    
    # Let's map these to U, V, W coefficients for a single 2x2x2 block.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # P1
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,1,1)]
    
    # P2
    u2 = [(1,0,1), (1,1,1)]
    v2 = [(0,0,1)]
    w2 = [(1,0,1), (1,1,-1)]
    
    # P3
    u3 = [(0,0,1)]
    v3 = [(0,1,1), (1,1,-1)]
    w3 = [(0,1,1), (1,1,1)]
    
    # P4
    u4 = [(1,1,1)]
    v4 = [(1,0,1), (0,0,-1)]
    w4 = [(0,0,1), (1,0,1)]
    
    # P5
    u5 = [(0,0,1), (0,1,1)]
    v5 = [(1,1,1)]
    w5 = [(0,0,-1), (0,1,1)]
    
    # P6
    u6 = [(1,0,1), (0,0,-1)]
    v6 = [(0,0,1), (0,1,1)]
    w6 = [(1,1,1)]
    
    # P7
    u7 = [(0,1,1), (1,1,-1)]
    v7 = [(1,0,1), (1,1,1)]
    w7 = [(0,0,1)]
    
    strassen_parts = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Iterate over the 8 blocks
    # j_offset in [0, 2]
    # k_offset in [0, 2, 4, 6]
    
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            for (us, vs, ws) in strassen_parts:
                # Add contributions to U
                for (i, j, val) in us:
                    global_i = i
                    global_j = j_off + j
                    U[global_i * 4 + global_j][col_idx] += val
                
                # Add contributions to V
                for (j, k, val) in vs:
                    global_j = j_off + j
                    global_k = k_off + k
                    V[global_j * 8 + global_k][col_idx] += val
                
                # Add contributions to W
                for (i, k, val) in ws:
                    global_i = i
                    global_k = k_off + k
                    W[global_k * 2 + global_i][col_idx] += val
                
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}
