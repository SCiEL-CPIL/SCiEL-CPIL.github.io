# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2
    # u_entries: list of (i, j, val)
    # v_entries: list of (j, k, val)
    # w_entries: list of (k, i, val)
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    # 1: M1 = (a00+a11)(b00+b11)
    add_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    
    # 2: M2 = (a10+a11)b00
    add_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)]
    )
    
    # 3: M3 = a00(b01-b11)
    add_term(
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)]
    )
    
    # 4: M4 = a11(b10-b00)
    add_term(
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)]
    )
    
    # 5: M5 = (a00+a01)b11
    add_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(1,0,1), (0,0,-1)]
    )
    
    # 6: M6 = (a10-a00)(b00+b01)
    add_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    )
    
    # 7: M7 = (a01-a11)(b10+b11)
    add_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    )
    
    # Fill matrices for two blocks (k=0,1 and k=2,3)
    for r in range(7):
        u_entries, v_entries, w_entries = terms[r]
        
        # Block 1 (k=0,1)
        col1 = r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col1] += val
            
        for j, k, val in v_entries:
            v_idx = j * p + k
            V[v_idx][col1] += val
            
        for k, i, val in w_entries:
            w_idx = k * n + i
            W[w_idx][col1] += val
            
        # Block 2 (k=2,3)
        col2 = 7 + r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col2] += val
            
        for j, k, val in v_entries:
            k_new = k + 2
            v_idx = j * p + k_new
            V[v_idx][col2] += val
            
        for k, i, val in w_entries:
            k_new = k + 2
            w_idx = k_new * n + i
            W[w_idx][col2] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
