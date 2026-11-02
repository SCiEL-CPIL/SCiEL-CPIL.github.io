import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 8 blocks of 2 columns each
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix), V (B matrix), W (C matrix)
    # Indices for U: 0: A00, 1: A01, 2: A10, 3: A11
    # Indices for V block: 0: B00, 1: B01, 2: B10, 3: B11
    # Indices for W block: 0: C00, 1: C01, 2: C10, 3: C11
    
    # M1: (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00+=1, C11+=1
    
    # M2: (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C10+=1, C11+=-1
    
    # M3: A00(B10+B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, 1]
    w3 = [0, 1, 0, 1] # C01+=1, C11+=1
    
    # M4: A11(B01-B11)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, 0, -1]
    w4 = [1, 0, 1, 0] # C00+=1, C10+=1
    
    # M5: (A00+A01)B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # C00+=-1, C01+=1
    
    # M6: (A10-A00)(B00+B01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11+=1
    
    # M7: (A01-A11)(B10-B00)
    u7 = [0, 1, 0, -1]
    v7 = [-1, 0, 1, 0]
    w7 = [1, 0, 0, 0] # C00+=1
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for l in range(num_blocks):
        for s in range(rank_per_block):
            col = l * rank_per_block + s
            u_coeff, v_coeff, w_coeff = terms[s]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_u = i * m + j
                    U[idx_u][col] = u_coeff[i * 2 + j]
            
            # Fill V
            # Block l corresponds to columns 2l, 2l+1
            # B00 is B[0, 2l], B01 is B[0, 2l+1]
            # B10 is B[1, 2l], B11 is B[1, 2l+1]
            # V index: j * p + k
            for i in range(2): # i is row of B (j in formula)
                for k in range(2): # k is col in block
                    idx_v = i * p + (2 * l + k)
                    # v_coeff indices: 0: B00, 1: B01, 2: B10, 3: B11
                    # i=0, k=0 -> B00 (0)
                    # i=0, k=1 -> B01 (1)
                    # i=1, k=0 -> B10 (2)
                    # i=1, k=1 -> B11 (3)
                    V[idx_v][col] = v_coeff[i * 2 + k]
            
            # Fill W
            # C00 is C[0, 2l], C01 is C[0, 2l+1]
            # C10 is C[1, 2l], C11 is C[1, 2l+1]
            # W index: k * n + i (k is col of C, i is row of C)
            # w_coeff indices: 0: C00, 1: C01, 2: C10, 3: C11
            for i in range(2): # i is row of C
                for k in range(2): # k is col in block
                    idx_w = (2 * l + k) * n + i
                    # w_coeff indices:
                    # k=0, i=0 -> C00 (0)
                    # k=1, i=0 -> C01 (1)
                    # k=0, i=1 -> C10 (2)
                    # k=1, i=1 -> C11 (3)
                    W[idx_w][col] = w_coeff[k * 2 + i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
