DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # We can just use schoolbook for the whole thing, which is 112.
    # But we can split.
    # However, the interface requires a single set of U, V, W.
    # We can just put the 64 terms in the first 64 columns.
    
    # Actually, the current program uses 112.
    # I will implement the split schoolbook.
    
    terms = []
    
    # Part 1: 4x4x4
    for i in range(4):
        for k in range(4):
            for j in range(4):
                a = i * 4 + k
                b = k * 7 + j
                c = j * 4 + i
                terms.append((a, b, c, 1))
                
    # Part 2: 4x4x3
    for i in range(4):
        for k in range(4):
            for j in range(3):
                a = i * 4 + k
                b = k * 7 + (4 + j)
                c = (4 + j) * 4 + i
                terms.append((a, b, c, 1))
                
    final_rank = len(terms)
    U = [[0] * final_rank for _ in range(n * m)]
    V = [[0] * final_rank for _ in range(m * p)]
    W = [[0] * final_rank for _ in range(p * n)]
    
    for c_idx, (a, b, ci, coeff) in enumerate(terms):
        U[a][c_idx] = coeff
        V[b][c_idx] = 1
        W[ci][c_idx] = 1
        
    return {"U": U, "V": V, "W": W}
