import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin
    
    n, m, p = 4, 4, 6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    # Define the 7 Strassen multipliers
    # Each: (list of (A_block_row, A_block_col, coeff)), 
    #        (list of (B_block_row, B_block_col, coeff)),
    #        (list of (C_block_row, C_block_col, coeff))
    
    strassen = [
        # M1 = (A11+A22)(B11+B22) -> C11(+), C22(+)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A21+A22)B11 -> C21(+), C22(-)
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A11(B12-B22) -> C12(+), C22(+)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A22(B21-B11) -> C11(+), C21(+)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A11+A12)B22 -> C11(-), C12(+)
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A21-A11)(B11+B12) -> C22(+)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22) -> C11(+)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen:
        for a in range(2):
            for b in range(2):
                for c in range(3):
                    for (abr, abc, ca) in a_blocks:
                        for (bbr, bbc, cb) in b_blocks:
                            for (cbr, cbc, cc) in c_blocks:
                                u_idx = (2*abr + a) * m + (2*abc + b)
                                v_idx = (2*bbr + b) * p + (3*bbc + c)
                                w_idx = (3*cbc + c) * n + (2*cbr + a)
                                coeff = ca * cb * cc
                                terms.append((u_idx, v_idx, w_idx, coeff))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][k] = coeff
        V[v_idx][k] = 1
        W[w_idx][k] = 1
    
    return {"U": U, "V": V, "W": W}
