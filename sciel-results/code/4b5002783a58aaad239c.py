"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # We use Strassen's algorithm for the 2x2 core.
    # There are p/m = 16/2 = 8 blocks.
    num_blocks = p // m
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]] -> indices (0,0), (0,1), (1,0), (1,1)
    # B = [[e, f], [g, h]] -> indices (0,0), (0,1), (1,0), (1,1)
    # C = [[c11, c12], [c21, c22]] -> indices (0,0), (0,1), (1,0), (1,1)
    
    # u_s: coeffs for A
    # v_s: coeffs for B
    # w_s: coeffs for C
    
    # M1 = a(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(e-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 + M5 + M7
    
    # Indexing:
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i (trace convention W[k*n+i])
    
    # For each block r in 0..7:
    # B columns are 2*r, 2*r+1
    # C columns are 2*r, 2*r+1
    
    for r in range(num_blocks):
        base_col = r * rank_per_block
        # B column offset
        b_col_offset = r * m
        
        for s in range(rank_per_block):
            col = base_col + s
            
            # Define u, v, w for this s
            # u is vector of length n*m = 4
            # v is vector of length m*p = 32
            # w is vector of length p*n = 32
            
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            if s == 0: # M1
                u[0] = 1 # a
                v[0 * m + b_col_offset] = 1 # e (j=0, k=2r)
                v[1 * m + b_col_offset + 1] = 1 # h (j=1, k=2r+1)
                # C11 (i=0, k=2r) -> w[2r * 2 + 0] = 1
                # C22 (i=1, k=2r+1) -> w[(2r+1) * 2 + 1] = 1
                w[b_col_offset * n + 0] = 1
                w[(b_col_offset + 1) * n + 1] = 1
                
            elif s == 1: # M2
                u[2] = 1 # c (i=1, j=0) -> idx 1*2+0=2
                u[3] = 1 # d (i=1, j=1) -> idx 1*2+1=3
                v[0 * m + b_col_offset] = 1 # e
                # C12 (i=0, k=2r+1) -> w[(2r+1) * 2 + 0] = 1
                w[(b_col_offset + 1) * n + 0] = 1
                
            elif s == 2: # M3
                u[0] = 1 # a
                v[0 * m + b_col_offset + 1] = 1 # f (j=0, k=2r+1)
                v[1 * m + b_col_offset + 1] = -1 # h
                # C21 (i=1, k=2r) -> w[2r * 2 + 1] = 1
                w[b_col_offset * n + 1] = 1
                
            elif s == 3: # M4
                u[3] = 1 # d
                v[0 * m + b_col_offset] = 1 # e
                v[0 * m + b_col_offset + 1] = -1 # f
                # C11 (i=0, k=2r) -> w[2r * 2 + 0] = -1
                # C12 (i=0, k=2r+1) -> w[(2r+1) * 2 + 0] = 1
                w[b_col_offset * n + 0] = -1
                w[(b_col_offset + 1) * n + 0] = 1
                
            elif s == 4: # M5
                u[0] = 1 # a
                u[1] = 1 # b
                v[1 * m + b_col_offset + 1] = 1 # h
                # C21 (i=1, k=2r) -> w[2r * 2 + 1] = 1
                # C22 (i=1, k=2r+1) -> w[(2r+1) * 2 + 1] = 1
                w[b_col_offset * n + 1] = 1
                w[(b_col_offset + 1) * n + 1] = 1
                
            elif s == 5: # M6
                u[2] = 1 # c
                u[0] = -1 # a
                v[0 * m + b_col_offset] = 1 # e
                v[0 * m + b_col_offset + 1] = 1 # f
                # C11 (i=0, k=2r) -> w[2r * 2 + 0] = 1
                w[b_col_offset * n + 0] = 1
                
            elif s == 6: # M7
                u[1] = 1 # b
                u[3] = -1 # d
                v[1 * m + b_col_offset] = 1 # g (j=1, k=2r)
                v[1 * m + b_col_offset + 1] = 1 # h
                # C11 (i=0, k=2r) -> w[2r * 2 + 0] = 1
                # C22 (i=1, k=2r+1) -> w[(2r+1) * 2 + 1] = 1
                w[b_col_offset * n + 0] = 1
                w[(b_col_offset + 1) * n + 1] = 1

            for idx in range(n * m):
                U[idx][col] = u[idx]
            for idx in range(m * p):
                V[idx][col] = v[idx]
            for idx in range(p * n):
                W[idx][col] = w[idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
