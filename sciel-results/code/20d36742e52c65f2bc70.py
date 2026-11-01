"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Block-diagonal Strassen decomposition: 5 independent 2x2 multiplications.
    # Total rank = 5 * 7 = 35.
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 5 blocks of 2 columns
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # U entries: rows are (i*m+j), i in {0,1}, j in {0,1} -> indices 0,1,2,3
    # V entries: rows are (j*p+k), j in {0,1}, k in {2t, 2t+1}
    # W entries: rows are (k*n+i), k in {2t, 2t+1}, i in {0,1}
    
    # Define the 7 Strassen products as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: [a, b, c, d] coefficients
    # V_coeffs: [e, f, g, h] coefficients for B_{0,2t}, B_{0,2t+1}, B_{1,2t}, B_{1,2t+1}
    # W_coeffs: [C11, C12, C21, C22] coefficients for output block
    
    strassen_products = [
        # (U, V, W)
        ([(1, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, 1), (0, 0, 1, 1)],
         [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],  # placeholder
         [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]),  # placeholder
    ]
    
    # Let's define them properly
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # U vectors (a,b,c,d):
    U_vecs = [
        (1, 0, 0, 1),  # P1: a+d
        (0, 0, 1, 1),  # P2: c+d
        (1, 0, 0, 0),  # P3: a
        (0, 0, 0, 1),  # P4: d
        (1, 1, 0, 0),  # P5: a+b
        (0, 0, 1, -1), # P6: c-a
        (0, 1, 0, -1), # P7: b-d
    ]
    
    # V vectors (e,f,g,h) where e=B_{0,2t}, f=B_{0,2t+1}, g=B_{1,2t}, h=B_{1,2t+1}
    V_vecs = [
        (1, 1, 0, 0),  # P1: e+h ... wait, P1=(a+d)(e+h) so V is (1,1,0,0) if h is last?
        # Let's fix: B is 2x2 block. B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h.
        # P1 = (a+d)(e+h) -> V=(1, 0, 0, 1)
        (1, 0, 0, 1),  # P1
        (1, 0, 0, 0),  # P2: e
        (0, 1, 0, -1), # P3: f-h
        (-1, 0, 1, 0), # P4: g-e
        (0, 0, 0, 1),  # P5: h
        (1, 1, 0, 0),  # P6: e+f
        (0, 0, 1, 1),  # P7: g+h
    ]
    
    # W vectors (coeffs for C11, C12, C21, C22)
    W_vecs = [
        (1, 0, 0, -1),  # P1 in C11, C22? No.
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # P1: C11=1, C12=0, C21=0, C22=1
        (1, 0, 0, 1),  # P1
        (0, 0, 1, -1), # P2: C21=1, C22=-1
        (0, 1, 0, 1),  # P3: C12=1, C22=1
        (1, 0, 1, 0),  # P4: C11=1, C21=1
        (-1, 1, 0, 0), # P5: C11=-1, C12=1
        (0, 0, 0, 1),  # P6: C22=1
        (1, 0, 0, 0),  # P7: C11=1
    ]
    
    for t in range(num_blocks):
        base_col = t * rank_per_block
        for r in range(rank_per_block):
            col = base_col + r
            u_vec = U_vecs[r]
            v_vec = V_vecs[r]
            w_vec = W_vecs[r]
            
            # Fill U
            for idx, val in enumerate(u_vec):
                if val != 0:
                    U[idx][col] = val
            
            # Fill V
            # V index = j * p + k
            # j=0, k=2t -> 0*p + 2t = 2t
            # j=0, k=2t+1 -> 2t+1
            # j=1, k=2t -> 1*p + 2t = 10 + 2t
            # j=1, k=2t+1 -> 10 + 2t + 1
            v_indices = [2*t, 2*t+1, 10+2*t, 10+2*t+1]
            for idx, val in enumerate(v_vec):
                if val != 0:
                    V[v_indices[idx]][col] = val
            
            # Fill W
            # W index = k * n + i
            # k=2t, i=0 -> 2t*2 + 0 = 4t
            # k=2t, i=1 -> 4t + 1
            # k=2t+1, i=0 -> (2t+1)*2 + 0 = 4t+2
            # k=2t+1, i=1 -> 4t+3
            w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
            for idx, val in enumerate(w_vec):
                if val != 0:
                    W[w_indices[idx]][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
