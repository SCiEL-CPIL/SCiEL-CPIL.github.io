def construct():
    import numpy as np
    
    n = 2
    m = 2
    p = 14
    num_blocks = p // m  # 7 blocks of 2x2 multiplication
    rank_per_block = 7   # Strassen rank for 2x2
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen formulas for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(h+f)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Indices for A (U): i*m + j
    # i=0, j=0 -> 0 (a)
    # i=0, j=1 -> 1 (b)
    # i=1, j=0 -> 2 (c)
    # i=1, j=1 -> 3 (d)
    
    # Indices for B (V): k*p + l
    # k=0, l=2b -> 2b (e)
    # k=0, l=2b+1 -> 2b+1 (f)
    # k=1, l=2b -> 14+2b (g)
    # k=1, l=2b+1 -> 14+2b+1 (h)
    
    # Indices for C (W): l*n + i
    # l=2b, i=0 -> 2b (C00)
    # l=2b, i=1 -> 2b+1 (C10)
    # l=2b+1, i=0 -> 2b+2 (C01)
    # l=2b+1, i=1 -> 2b+3 (C11)
    
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        base_col = 2 * b
        
        # M1 = (a+d)(e+h)
        r = base_rank + 0
        U[0, r] += 1; U[3, r] += 1  # a+d
        V[base_col, r] += 1; V[14 + base_col, r] += 1  # e+h
        W[2*base_col, r] += 1; W[2*base_col+1, r] += 1; W[2*base_col+2, r] -= 1; W[2*base_col+3, r] += 1  # C00+1, C10+1, C01-1, C11+1
        
        # M2 = (c+d)e
        r = base_rank + 1
        U[2, r] += 1; U[3, r] += 1  # c+d
        V[base_col, r] += 1  # e
        W[2*base_col+1, r] += 1; W[2*base_col+3, r] -= 1  # C10+1, C11-1
        
        # M3 = a(g-h)
        r = base_rank + 2
        U[0, r] += 1  # a
        V[14 + base_col, r] += 1; V[14 + base_col + 1, r] -= 1  # g-h
        W[2*base_col+2, r] += 1; W[2*base_col+3, r] += 1  # C01+1, C11+1
        
        # M4 = d(f-e)
        r = base_rank + 3
        U[3, r] += 1  # d
        V[base_col + 1, r] += 1; V[base_col, r] -= 1  # f-e
        W[2*base_col, r] += 1; W[2*base_col+1, r] += 1  # C00+1, C10+1
        
        # M5 = (a+b)h
        r = base_rank + 4
        U[0, r] += 1; U[1, r] += 1  # a+b
        V[14 + base_col + 1, r] += 1  # h
        W[2*base_col, r] -= 1; W[2*base_col+2, r] += 1  # C00-1, C01+1
        
        # M6 = (c-a)(e+g)
        r = base_rank + 5
        U[2, r] += 1; U[0, r] -= 1  # c-a
        V[base_col, r] += 1; V[14 + base_col, r] += 1  # e+g
        W[2*base_col+3, r] += 1  # C11+1
        
        # M7 = (b-d)(h+f)
        r = base_rank + 6
        U[1, r] += 1; U[3, r] -= 1  # b-d
        V[14 + base_col + 1, r] += 1; V[base_col + 1, r] += 1  # h+f
        W[2*base_col, r] += 1  # C00+1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
