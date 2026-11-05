import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_222(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        # M1 = (A00+A11)*(B00+B11) -> C00 + C11
        add_term({(a_row_base+0)*m + (a_col_base+0): 1, (a_row_base+1)*m + (a_col_base+1): 1},
                 {(b_row_base+0)*p + (b_col_base+0): 1, (b_row_base+1)*p + (b_col_base+1): 1},
                 {(c_col_base+0)*n + (c_row_base+0): 1, (c_col_base+1)*n + (c_row_base+1): 1})
        
        # M2 = (A10+A11)*B00 -> C10
        add_term({(a_row_base+1)*m + (a_col_base+0): 1, (a_row_base+1)*m + (a_col_base+1): 1},
                 {(b_row_base+0)*p + (b_col_base+0): 1},
                 {(c_col_base+0)*n + (c_row_base+1): 1})
        
        # M3 = A00*(B01-B11) -> C01 + C10
        add_term({(a_row_base+0)*m + (a_col_base+0): 1},
                 {(b_row_base+0)*p + (b_col_base+1): 1, (b_row_base+1)*p + (b_col_base+1): -1},
                 {(c_col_base+1)*n + (c_row_base+0): 1, (c_col_base+0)*n + (c_row_base+1): 1})
        
        # M4 = A11*(B10-B00) -> C10 + C11
        add_term({(a_row_base+1)*m + (a_col_base+1): 1},
                 {(b_row_base+1)*p + (b_col_base+0): 1, (b_row_base+0)*p + (b_col_base+0): -1},
                 {(c_col_base+0)*n + (c_row_base+1): 1, (c_col_base+1)*n + (c_row_base+1): 1})
        
        # M5 = (A00+A01)*B11 -> C01 - C11
        add_term({(a_row_base+0)*m + (a_col_base+0): 1, (a_row_base+0)*m + (a_col_base+1): 1},
                 {(b_row_base+1)*p + (b_col_base+1): 1},
                 {(c_col_base+1)*n + (c_row_base+0): 1, (c_col_base+1)*n + (c_row_base+1): -1})
        
        # M6 = (A10-A00)*(B00+B01) -> C11
        add_term({(a_row_base+1)*m + (a_col_base+0): 1, (a_row_base+0)*m + (a_col_base+0): -1},
                 {(b_row_base+0)*p + (b_col_base+0): 1, (b_row_base+0)*p + (b_col_base+1): 1},
                 {(c_col_base+1)*n + (c_row_base+1): 1})
        
        # M7 = (A01-A11)*(B10+B11) -> C00
        add_term({(a_row_base+0)*m + (a_col_base+1): 1, (a_row_base+1)*m + (a_col_base+1): -1},
                 {(b_row_base+1)*p + (b_col_base+0): 1, (b_row_base+1)*p + (b_col_base+1): 1},
                 {(c_col_base+0)*n + (c_row_base+0): 1})
    
    def schoolbook_122(a_row, a_col_base, b_row_base, b_col_base, c_row, c_col_base):
        for j in range(2):
            for k in range(2):
                add_term({a_row*m + (a_col_base+j): 1},
                         {(b_row_base+j)*p + (b_col_base+k): 1},
                         {(c_col_base+k)*n + c_row: 1})
    
    # C_00 = A_00 * B_00 + A_01 * B_10
    strassen_222(0, 0, 0, 0, 0, 0)
    strassen_222(0, 2, 2, 0, 0, 0)
    
    # C_01 = A_00 * B_01 + A_01 * B_11
    strassen_222(0, 0, 0, 2, 0, 2)
    strassen_222(0, 2, 2, 2, 0, 2)
    
    # C_10 = A_10 * B_00 + A_11 * B_10
    schoolbook_122(2, 0, 0, 0, 2, 0)
    schoolbook_122(2, 2, 2, 0, 2, 0)
    
    # C_11 = A_10 * B_01 + A_11 * B_11
    schoolbook_122(2, 0, 0, 2, 2, 2)
    schoolbook_122(2, 2, 2, 2, 2, 2)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][c] = coeff
        for idx, coeff in v.items():
            V[idx][c] = coeff
        for idx, coeff in w.items():
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
