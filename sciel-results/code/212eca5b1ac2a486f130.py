import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_i, u_k, v_k, v_j, w_j, w_i, coeff):
        nonlocal col
        u_idx = u_i * m + u_k
        v_idx = v_k * p + v_j
        w_idx = w_j * n + w_i
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen vectors for 2x2x2
    # u: A[0,0], A[0,1], A[1,0], A[1,1]
    # v: B[0,0], B[0,1], B[1,0], B[1,1]
    # w: C[0,0], C[0,1], C[1,0], C[1,1] -> indices (c,a) -> (0,0), (1,0), (0,1), (1,1)
    # Note: w index in global is j*n + i. Local j is c, local i is a.
    
    # M1
    strassen = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: (0,0):1, (1,1):1
        (({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})),
        # M2
        (({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})),
        # M3
        (({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (1,1):1})),
        # M4
        (({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (0,1):1})),
        # M5
        (({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})),
        # M6
        (({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1})),
        # M7
        (({(0,1):1, (1,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1})),
    ]

    # Process 3 blocks of 2 columns (0-1, 2-3, 4-5)
    for block in range(3):
        j_offset = block * 2
        
        # Top 2 rows of C (i=0,1)
        # Part 1: Strassen for k=0,1
        for u_dict, v_dict, w_dict in strassen:
            # Expand dicts into terms
            # We need to add terms for each combination?
            # No, the Strassen term is a single rank-1 term in the 2x2x2 space?
            # Wait. The Strassen algorithm provides 7 rank-1 terms for the 2x2x2 tensor.
            # So we add 7 terms here.
            # The vectors are sparse.
            # u_dict maps (a, b) -> coeff
            # v_dict maps (b, c) -> coeff
            # w_dict maps (c, a) -> coeff
            
            # But a single rank-1 term requires u, v, w to be vectors.
            # The "term" in the sum is $u \otimes v \otimes w$.
            # If u has two non-zero entries, it's not a basis vector.
            # It is a valid vector.
            # So we just add one term for each of the 7 Strassen formulas.
            
            # u vector
            u_vec = np.zeros(4)
            for (a, b), c_u in u_dict.items():
                u_vec[a*2 + b] = c_u
            # v vector
            v_vec = np.zeros(4)
            for (b, c), c_v in v_dict.items():
                v_vec[b*2 + c] = c_v
            # w vector
            w_vec = np.zeros(4)
            for (c, a), c_w in w_dict.items():
                w_vec[c*2 + a] = c_w
                
            # Now add these vectors to U, V, W
            # Map local indices to global
            # u: a in {0,1}, b in {0,1} -> i in {0,1}, k in {0,1}
            for local_u_idx, val in enumerate(u_vec):
                if val != 0:
                    a, b = divmod(local_u_idx, 2)
                    add_term(a, b, -1, -1, -1, -1, 0) # Dummy, we need to add to specific col
                    # Wait, add_term increments col.
                    # We should add all components of this rank-1 term to the SAME column index.
                    pass
            
            # Let's do it manually for one column
            col_idx = col
            col += 1
            
            for local_u_idx, val in enumerate(u_vec):
                if val != 0:
                    a, b = divmod(local_u_idx, 2)
                    u_idx = a * m + b
                    U[u_idx][col_idx] = val
            
            for local_v_idx, val in enumerate(v_vec):
                if val != 0:
                    b, c = divmod(local_v_idx, 2)
                    # b is local k, c is local j
                    global_k = b
                    global_j = j_offset + c
                    v_idx = global_k * p + global_j
                    V[v_idx][col_idx] = val
            
            for local_w_idx, val in enumerate(w_vec):
                if val != 0:
                    c, a = divmod(local_w_idx, 2)
                    # c is local j, a is local i
                    global_j = j_offset + c
                    global_i = a
                    w_idx = global_j * n + global_i
                    W[w_idx][col_idx] = val

        # Part 2: Schoolbook for k=2
        # A[0,2] * B[2, j0] -> C[0, j0]
        add_term(0, 2, 2, j_offset + 0, j_offset + 0, 0, 1)
        # A[0,2] * B[2, j1] -> C[0, j1]
        add_term(0, 2, 2, j_offset + 1, j_offset + 1, 0, 1)
        # A[1,2] * B[2, j0] -> C[1, j0]
        add_term(1, 2, 2, j_offset + 0, j_offset + 0, 1, 1)
        # A[1,2] * B[2, j1] -> C[1, j1]
        add_term(1, 2, 2, j_offset + 1, j_offset + 1, 1, 1)

        # Bottom 1 row of C (i=2)
        # Part 1: Schoolbook for k=0,1
        # A[2,0] * B[0, j0] -> C[2, j0]
        add_term(2, 0, 0, j_offset + 0, j_offset + 0, 2, 1)
        # A[2,0] * B[0, j1] -> C[2, j1]
        add_term(2, 0, 0, j_offset + 1, j_offset + 1, 2, 1)
        # A[2,1] * B[1, j0] -> C[2, j0]
        add_term(2, 1, 1, j_offset + 0, j_offset + 0, 2, 1)
        # A[2,1] * B[1, j1] -> C[2, j1]
        add_term(2, 1, 1, j_offset + 1, j_offset + 1, 2, 1)
        
        # Part 2: Schoolbook for k=2
        # A[2,2] * B[2, j0] -> C[2, j0]
        add_term(2, 2, 2, j_offset + 0, j_offset + 0, 2, 1)
        # A[2,2] * B[2, j1] -> C[2, j1]
        add_term(2, 2, 2, j_offset + 1, j_offset + 1, 2, 1)

    # Last block: 1 column (6)
    j_offset = 6
    
    # Top 2 rows
    # k=0,1 Schoolbook (Rank 4)
    add_term(0, 0, 0, j_offset, j_offset, 0, 1)
    add_term(0, 1, 1, j_offset, j_offset, 0, 1)
    add_term(1, 0, 0, j_offset, j_offset, 1, 1)
    add_term(1, 1, 1, j_offset, j_offset, 1, 1)
    
    # k=2 Schoolbook (Rank 2)
    add_term(0, 2, 2, j_offset, j_offset, 0, 1)
    add_term(1, 2, 2, j_offset, j_offset, 1, 1)
    
    # Bottom 1 row
    # k=0,1 Schoolbook (Rank 2)
    add_term(2, 0, 0, j_offset, j_offset, 2, 1)
    add_term(2, 1, 1, j_offset, j_offset, 2, 1)
    
    # k=2 Schoolbook (Rank 1)
    add_term(2, 2, 2, j_offset, j_offset, 2, 1)

    return {"U": U, "V": V, "W": W}
