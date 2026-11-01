"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_cols, b_rows, c_cols, base_col):
        col = base_col
        a0, a1 = a_cols
        b0, b1 = b_rows
        c0, c1 = c_cols
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[0*4 + a0][col] += 1
        U[1*4 + a1][col] += 1
        V[b0*4 + c0][col] += 1
        V[b1*4 + c1][col] += 1
        W[c0*2 + 0][col] += 1
        W[c1*2 + 1][col] += 1
        col += 1
        
        # M2 = (X10 + X11)Y00
        U[1*4 + a0][col] += 1
        U[1*4 + a1][col] += 1
        V[b0*4 + c0][col] += 1
        W[c0*2 + 1][col] += 1
        col += 1
        
        # M3 = X00(Y01 - Y11)
        U[0*4 + a0][col] += 1
        V[b0*4 + c1][col] += 1
        V[b1*4 + c1][col] -= 1
        W[c1*2 + 0][col] += 1
        col += 1
        
        # M4 = X11(Y10 - Y00)
        U[1*4 + a1][col] += 1
        V[b1*4 + c0][col] += 1
        V[b0*4 + c0][col] -= 1
        W[c0*2 + 0][col] += 1
        col += 1
        
        # M5 = (X00 + X10)Y11
        U[0*4 + a0][col] += 1
        U[1*4 + a0][col] += 1
        V[b1*4 + c1][col] += 1
        W[c0*2 + 0][col] -= 1
        W[c1*2 + 0][col] += 1
        col += 1
        
        # M6 = (X01 - X00)(Y00 + Y10)
        U[0*4 + a1][col] += 1
        U[0*4 + a0][col] -= 1
        V[b0*4 + c0][col] += 1
        V[b1*4 + c0][col] += 1
        W[c1*2 + 1][col] += 1
        col += 1
        
        # M7 = (X10 - X11)(Y01 + Y11)
        U[1*4 + a0][col] += 1
        U[1*4 + a1][col] -= 1
        V[b0*4 + c1][col] += 1
        V[b1*4 + c1][col] += 1
        W[c0*2 + 0][col] += 1
        col += 1
        
        return col

    # 1. A_top * B_TL -> C_left
    col = add_strassen_block([0, 1], [0, 1], [0, 1], 0)
    
    # 2. A_bot * B_BL -> C_left
    col = add_strassen_block([2, 3], [2, 3], [0, 1], col)
    
    # 3. A_top * B_TR -> C_right
    col = add_strassen_block([0, 1], [0, 1], [2, 3], col)
    
    # 4. A_bot * B_BR -> C_right
    col = add_strassen_block([2, 3], [2, 3], [2, 3], col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
