"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # Local indices for 2x2 matrices (row-major): 0:00, 1:01, 2:10, 3:11
    # Each term is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    strassen_terms = [
        # P1: (x00+x11)(y00+y11) -> z00+1, z11+1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2: (x10+x11)y00 -> z10+1, z11-1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # P3: x00(y10-y11) -> z01+1, z11+1
        ([0], [1], [2, 3], [1, -1], [1, 3], [1, 1]),
        # P4: x11(y01-y00) -> z00+1, z10+1
        ([3], [1], [1, 0], [1, -1], [0, 2], [1, 1]),
        # P5: (x00+x01)y11 -> z00-1, z01+1
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # P6: (x10-x00)(y00+y01) -> z11+1
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # P7: (x01-x11)(y10+y11) -> z00+1
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # Fill the 3 Strassen blocks (21 columns)
    for t in range(3):
        for s in range(7):
            col = t * 7 + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_terms[s]
            
            for ui, uv in zip(u_idx, u_val):
                i = ui // 2
                j = ui % 2
                global_u = i * m + j
                U[global_u][col] += uv
                
            for vi, vv in zip(v_idx, v_val):
                i = vi // 2
                j = vi % 2
                global_v = i * p + (2 * t + j)
                V[global_v][col] += vv
                
            for wi, wv in zip(w_idx, w_val):
                i = wi // 2
                j = wi % 2
                global_w = (2 * t + j) * n + i
                W[global_w][col] += wv
                
    # Fill the tail part (12 columns)
    # A_{2x1} * B_{1x6}
    for i in range(n):
        for k in range(p):
            col = 21 + i * p + k
            global_u = i * m + (m - 1)
            global_v = (m - 1) * p + k
            global_w = k * n + i
            
            U[global_u][col] = 1
            V[global_v][col] = 1
            W[global_w][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
