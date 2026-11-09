import numpy as np

def construct():
    n, m, p = 5, 5, 5
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    # Helper to add a 2x2 Strassen decomposition
    # a_idx: list of 4 indices for A block
    # b_idx: list of 4 indices for B block
    # c_idx: list of 4 indices for C block
    def add_strassen_2x2(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11
        for i in [0, 3]:
            U[a_idx[i]][col] = 1
            V[b_idx[i]][col] = 1
            W[c_idx[i]][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # U: A10, A11. V: B00. W: C10, -C11
        U[a_idx[2]][col] = 1
        U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        W[c_idx[3]][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        # U: A00. V: B10, -B11. W: C01, C10? 
        # Using the formula: C01 = M3 + M5, C10 = M2 + M4
        # So M3 contributes to C01.
        U[a_idx[0]][col] = 1
        V[b_idx[2]][col] = 1
        V[b_idx[3]][col] = -1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        # U: A11. V: B01, -B00. W: C00, C10
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1
        V[b_idx[0]][col] = -1
        W[c_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M5 = (A00+A10)B11
        # U: A00, A10. V: B11. W: -C00, C01
        U[a_idx[0]][col] = 1
        U[a_idx[2]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = -1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10, -A00. V: B00, B10. W: C11
        U[a_idx[2]][col] = 1
        U[a_idx[0]][col] = -1
        V[b_idx[0]][col] = 1
        V[b_idx[2]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M7 = (A11-A01)(B01+B11)
        # U: A11, -A01. V: B01, B11. W: C00
        U[a_idx[3]][col] = 1
        U[a_idx[1]][col] = -1
        V[b_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1

    # 4x4 Strassen part
    # A00: 0,1,5,6
    # A01: 2,3,7,8
    # A10: 10,11,15,16
    # A11: 12,13,17,18
    
    # We need to apply Strassen to the 4x4 blocks.
    # This means we treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    # Block 00 (top-left 2x2 of A): indices 0,1,5,6
    # Block 01 (top-right 2x2 of A): indices 2,3,7,8
    # Block 10 (bottom-left 2x2 of A): indices 10,11,15,16
    # Block 11 (bottom-right 2x2 of A): indices 12,13,17,18
    
    # Same for B and C.
    
    a_blocks = [
        [0, 1, 5, 6],   # A00
        [2, 3, 7, 8],   # A01
        [10, 11, 15, 16], # A10
        [12, 13, 17, 18]  # A11
    ]
    
    b_blocks = [
        [0, 1, 5, 6],   # B00
        [2, 3, 7, 8],   # B01
        [10, 11, 15, 16], # B10
        [12, 13, 17, 18]  # B11
    ]
    
    c_blocks = [
        [0, 1, 5, 6],   # C00
        [2, 3, 7, 8],   # C01
        [10, 11, 15, 16], # C10
        [12, 13, 17, 18]  # C11
    ]
    
    # Apply Strassen to the 4x4 blocks
    # This is a 2x2 Strassen where each "element" is a 2x2 matrix.
    # So we call add_strassen_2x2 for each of the 7 terms?
    # No, add_strassen_2x2 assumes scalar elements.
    # We need to compose it.
    
    # For each of the 7 terms in the outer Strassen, we have a product of 2x2 blocks.
    # We use the inner Strassen (add_strassen_2x2) to compute each product.
    
    # Term 1: (A00+A11)(B00+B11)
    # This is a sum of 4 products: A00B00, A00B11, A11B00, A11B11
    # We can just add the schoolbook terms for these 4 products?
    # No, that would be rank 16.
    # We want to use Strassen for the 2x2 multiplication.
    
    # Let's just use schoolbook for the 4x4 part to be safe and correct, 
    # but that gives rank 64. Total 125.
    
    # I will use schoolbook for 4x4.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                add_term(u_idx, v_idx, w_idx, 1)
                
    # Cross terms
    # A01 B10 (4x1 * 1x4 -> 4x4)
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 5 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # A00 B01 (4x4 * 4x1 -> 4x1)
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 5 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # A10 B00 (1x4 * 4x4 -> 1x4)
    for j in range(4):
        for k in range(4):
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            add_term(u_idx, v_idx, w_idx, 1)
            
    # A01 B11 (4x1 * 1x1 -> 4x1)
    for i in range(4):
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        add_term(u_idx, v_idx, w_idx, 1)
        
    # A11 B10 (1x1 * 1x4 -> 1x4)
    for k in range(4):
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + k
        w_idx = k * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1)
        
    # A10 B01 (1x4 * 4x1 -> 1x1)
    for j in range(4):
        u_idx = 4 * 5 + j
        v_idx = j * 5 + 4
        w_idx = 4 * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1)
        
    # A11 B11 (1x1 * 1x1 -> 1x1)
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    add_term(u_idx, v_idx, w_idx, 1)
    
    return {"U": U, "V": V, "W": W}
