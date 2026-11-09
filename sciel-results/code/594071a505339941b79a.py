def construct():
    n, m, p = DIMENSIONS
    
    U = [[0] * 100 for _ in range(n * m)] # Pre-allocate enough, or use list of dicts
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    # Better: use a list of columns
    cols = []
    
    def add_column(u_dict, v_dict, w_dict):
        cols.append((u_dict, v_dict, w_dict))
        
    # ... logic to add columns ...
    
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(cols):
        for u_idx, coeff in u_dict.items():
            U[u_idx][c] = coeff
        for v_idx, coeff in v_dict.items():
            V[v_idx][c] = coeff
        for w_idx, coeff in w_dict.items():
            W[w_idx][c] = coeff
            
    return {"U": U, "V": V, "W": W}
