DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # 7 independent 2x2x2 multiplications
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    for b in range(num_blocks):
        v_off = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_off = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_d, v_d, w_d in terms:
            for idx, val in u_d.items():
                U[idx][col] += val
            for li, val in v_d.items():
                V[v_off[li]][col] += val
            for li, val in w_d.items():
                W[w_off[li]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
