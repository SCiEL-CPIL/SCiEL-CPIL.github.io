"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    # Rank 39 decomposition
    # U: 6 x R, V: 21 x R, W: 14 x R
    # We build lists of (U_idx, val), (V_idx, val), (W_idx, val) for each rank term
    terms = []
    
    # Part 1: A1 B1 (2x1 * 1x7) -> Rank 14
    # A1 is col 0 of A. B1 is row 0 of B.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            terms.append([(u_idx, 1), (v_idx, 1), (w_idx, 1)])
            
    # Part 2: A2 B2 (2x2 * 2x7)
    # A2 is cols 1,2 of A. B2 is rows 1,2 of B.
    # Split B2 into blocks of 2 cols, and one block of 1 col.
    # B2 cols: 0,1 -> Block 0; 2,3 -> Block 1; 4,5 -> Block 2; 6 -> Block 3
    
    # Strassen coefficients for 2x2
    # A2 = [[a00, a01], [a10, a11]]
    # B2 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U indices for A2:
    # a00 -> i=0, j=1 -> 0*3+1 = 1
    # a01 -> i=0, j=2 -> 0*3+2 = 2
    # a10 -> i=1, j=1 -> 1*3+1 = 4
    # a11 -> i=1, j=2 -> 1*3+2 = 5
    
    # V indices for B2 block at cols k1, k2:
    # b00 -> row 1, col k1 -> 1*7+k1
    # b01 -> row 1, col k2 -> 1*7+k2
    # b10 -> row 2, col k1 -> 2*7+k1
    # b11 -> row 2, col k2 -> 2*7+k2
    
    # W indices for C block at cols k1, k2:
    # C00 -> k1*n+0
    # C01 -> k2*n+0
    # C10 -> k1*n+1
    # C11 -> k2*n+1
    
    def add_strassen_block(k1, k2):
        # b00, b01, b10, b11 indices
        v_b00 = 1 * p + k1
        v_b01 = 1 * p + k2
        v_b10 = 2 * p + k1
        v_b11 = 2 * p + k2
        
        # w indices
        w_c00 = k1 * n + 0
        w_c01 = k2 * n + 0
        w_c10 = k1 * n + 1
        w_c11 = k2 * n + 1
        
        # u indices
        u_a00 = 0 * m + 1
        u_a01 = 0 * m + 2
        u_a10 = 1 * m + 1
        u_a11 = 1 * m + 2
        
        # M1
        terms.append([(u_a00, 1), (u_a11, 1), (v_b00, 1), (v_b11, 1), (w_c00, 1), (w_c11, 1)])
        # Wait, terms are rank 1 tensors. Each term is a product of one U vec, one V vec, one W vec.
        # My 'terms' list above was adding multiple entries to one term?
        # No, a rank 1 term is (u_vec, v_vec, w_vec).
        # I need to represent u_vec, v_vec, w_vec as sparse dicts or similar.
        
    # Let's restart the term construction properly.
    # Each term is a tuple (U_dict, V_dict, W_dict)
    
    terms = []
    
    # Part 1: A1 B1
    for i in range(n):
        for k in range(p):
            u = {i * m + 0: 1}
            v = {0 * p + k: 1}
            w = {k * n + i: 1}
            terms.append((u, v, w))
            
    # Part 2: A2 B2
    # Strassen blocks
    blocks = [(0, 1), (2, 3), (4, 5)]
    
    def get_strassen_terms(k1, k2):
        v_b00 = 1 * p + k1
        v_b01 = 1 * p + k2
        v_b10 = 2 * p + k1
        v_b11 = 2 * p + k2
        
        w_c00 = k1 * n + 0
        w_c01 = k2 * n + 0
        w_c10 = k1 * n + 1
        w_c11 = k2 * n + 1
        
        u_a00 = 0 * m + 1
        u_a01 = 0 * m + 2
        u_a10 = 1 * m + 1
        u_a11 = 1 * m + 2
        
        res = []
        
        # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        u = {u_a00: 1, u_a11: 1}
        v = {v_b00: 1, v_b11: 1}
        w = {w_c00: 1, w_c11: 1}
        res.append((u, v, w))
        
        # M2 = (a10+a11)b00 -> C10 += 1, C11 -= 1
        u = {u_a10: 1, u_a11: 1}
        v = {v_b00: 1}
        w = {w_c10: 1, w_c11: -1}
        res.append((u, v, w))
        
        # M3 = a00(b01-b11) -> C01 += 1, C11 += 1
        u = {u_a00: 1}
        v = {v_b01: 1, v_b11: -1}
        w = {w_c01: 1, w_c11: 1}
        res.append((u, v, w))
        
        # M4 = a11(b10-b00) -> C00 -= 1, C10 += 1
        u = {u_a11: 1}
        v = {v_b10: 1, v_b00: -1}
        w = {w_c00: -1, w_c10: 1}
        res.append((u, v, w))
        
        # M5 = (a00+a01)b11 -> C00 -= 1, C01 += 1
        u = {u_a00: 1, u_a01: 1}
        v = {v_b11: 1}
        w = {w_c00: -1, w_c01: 1}
        res.append((u, v, w))
        
        # M6 = (a10-a00)(b00+b01) -> C11 += 1
        u = {u_a10: 1, u_a00: -1}
        v = {v_b00: 1, v_b01: 1}
        w = {w_c11: 1}
        res.append((u, v, w))
        
        # M7 = (a01-a11)(b10+b11) -> C00 += 1
        u = {u_a01: 1, u_a11: -1}
        v = {v_b10: 1, v_b11: 1}
        w = {w_c00: 1}
        res.append((u, v, w))
        
        return res

    for k1, k2 in blocks:
        terms.extend(get_strassen_terms(k1, k2))
        
    # Block 3: k1=6, k2=None (2x1)
    # A2 (2x2) * B24 (2x1)
    # B24 = [[b00], [b10]]
    # C = [[a00 b00 + a01 b10], [a10 b00 + a11 b10]]
    # Rank 4 terms:
    # a00 b00 -> C00
    # a01 b10 -> C00
    # a10 b00 -> C10
    # a11 b10 -> C10
    
    k1 = 6
    v_b00 = 1 * p + k1
    v_b10 = 2 * p + k1
    w_c00 = k1 * n + 0
    w_c10 = k1 * n + 1
    
    u_a00 = 0 * m + 1
    u_a01 = 0 * m + 2
    u_a10 = 1 * m + 1
    u_a11 = 1 * m + 2
    
    terms.append(({u_a00: 1}, {v_b00: 1}, {w_c00: 1}))
    terms.append(({u_a01: 1}, {v_b10: 1}, {w_c00: 1}))
    terms.append(({u_a10: 1}, {v_b00: 1}, {w_c10: 1}))
    terms.append(({u_a11: 1}, {v_b10: 1}, {w_c10: 1}))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][c] = val
        for idx, val in v.items():
            V[idx][c] = val
        for idx, val in w.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
