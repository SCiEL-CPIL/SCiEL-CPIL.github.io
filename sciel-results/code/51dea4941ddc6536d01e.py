"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        u_coeffs = [
            [1, 0, 0, 1], 
            [0, 0, 1, 1], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0], 
            [0, 1, 0, -1]  
        ]
        
        v_coeffs = [
            [1, 0, 0, 1], 
            [1, 0, 0, 0], 
            [0, -1, 1, 0], 
            [0, 1, 0, -1], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [0, 0, 1, 1]  
        ]
        
        w_coeffs = [
            [1, 0, 0, 1], 
            [0, 0, 1, -1], 
            [0, 1, 0, 1], 
            [1, 0, 1, 0], 
            [-1, 1, 0, 0], 
            [0, 0, 0, 1], 
            [1, -1, 0, 0]  
        ]
        
        for r in range(7):
            c_idx = col + r
            for lr in range(2):
                for lc in range(2):
                    val = u_coeffs[r][lr * 2 + lc]
                    if val != 0:
                        U[a_rows[lr] * m + a_cols[lc]][c_idx] += val
            for lr in range(2):
                for lc in range(2):
                    val = v_coeffs[r][lr * 2 + lc]
                    if val != 0:
                        V[b_rows[lr] * p + b_cols[lc]][c_idx] += val
            for lr in range(2):
                for lc in range(2):
                    val = w_coeffs[r][lr * 2 + lc]
                    if val != 0:
                        W[c_cols[lc] * n + c_rows[lr]][c_idx] += val
        col += 7

    def add_schoolbook_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        nonlocal col
        terms = [
            (0, 0, 0, 0), 
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1)
        ]
        
        for r in range(4):
            c_idx = col + r
            ar, ac, br, cr = terms[r]
            U[a_rows[ar] * m + a_cols[ac]][c_idx] = 1
            V[b_rows[br] * p + b_col][c_idx] = 1
            W[c_col * n + c_rows[cr]][c_idx] = 1
        col += 4

    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_schoolbook_2x2x1([0, 1], [0, 1], [0, 1], 4, [0, 1], 4)
    
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    add_schoolbook_2x2x1([0, 1], [2, 3], [2, 3], 4, [0, 1], 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
