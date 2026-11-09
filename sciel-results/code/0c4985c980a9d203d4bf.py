"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tiling 6x6 into 3x3 blocks of 2x2
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # Total blocks: 3 (rows of A) * 3 (cols of A) * 3 (rows of B) = 27 block products
    # Total rank: 27 * 7 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # Local indices:
    # U: u*2 + v (u=row, v=col)
    # V: v*2 + w (v=row, w=col)
    # W: w*2 + u (w=col, u=row)
    
    # s=1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # U: 0,3; V: 0,3; W: 0,3
    # s=2: (A10+A11)B00 -> C10+=1
    # U: 2,3; V: 0; W: 1
    # s=3: A00(B01-B11) -> C01+=1
    # U: 0; V: 1,3 (1,-1); W: 2
    # s=4: A11(B10-B00) -> C00+=1, C10+=1
    # U: 3; V: 2,0 (1,-1); W: 0,1
    # s=5: (A00+A01)B11 -> C00-=1, C01+=1
    # U: 0,1; V: 3; W: 0,-2? No W: 0,2 with signs -1, 1
    # s=6: (A10-A00)(B00+B01) -> C11+=1
    # U: 2,0 (1,-1); V: 0,1; W: 3
    # s=7: (A01-A11)(B10+B11) -> C00+=1
    # U: 1,3 (1,-1); V: 2,3; W: 0
    
    strassen_factors = [
        # (U_indices, U_values, V_indices, V_values, W_indices, W_values)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # s=1
        ([2, 3], [1, 1], [0], [1], [1], [1]),             # s=2
        ([0], [1], [1, 3], [1, -1], [2], [1]),            # s=3
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),      # s=4
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),      # s=5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),      # s=6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])       # s=7
    ]
    
    col = 0
    for r_A in range(3):
        for c in range(3):
            for r_B in range(3):
                for s in range(7):
                    u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_factors[s]
                    
                    for idx, val in zip(u_idx, u_val):
                        u, v = divmod(idx, 2)
                        global_u = 2 * r_A + u
                        global_v = 2 * c + v
                        U[global_u * 6 + global_v][col] = val
                    
                    for idx, val in zip(v_idx, v_val):
                        v, w = divmod(idx, 2)
                        global_v = 2 * c + v
                        global_w = 2 * r_B + w
                        V[global_v * 6 + global_w][col] = val
                    
                    for idx, val in zip(w_idx, w_val):
                        w, u = divmod(idx, 2)
                        global_w = 2 * r_B + w
                        global_u = 2 * r_A + u
                        W[global_w * 6 + global_u][col] = val
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
