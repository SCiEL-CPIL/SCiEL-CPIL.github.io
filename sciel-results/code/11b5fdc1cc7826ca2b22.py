"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Use 6 blocks of 2x2x2 matrix multiplication (Strassen's algorithm)
    # Dimensions: n=2, m=2, p=12
    # p=12 is 6 blocks of p=2
    # Rank of 2x2x2 is 7
    # Total rank = 6 * 7 = 42
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Indices for A (2x2): 0: a, 1: b, 2: c, 3: d
    # Indices for B (2x2): 0: e, 1: f, 2: g, 3: h
    # Indices for C (2x2): 0: C11, 1: C12, 2: C21, 3: C22
    
    # Strassen terms (u, v, w)
    # u is for A, v is for B, w is for C
    # Each term is a triple of vectors.
    # We define the coefficients for each of the 7 multiplications.
    
    strassen_terms = [
        # m1 = (a+d)(e+h)
        # u: a=1, d=1 => [1, 0, 0, 1]
        # v: e=1, h=1 => [1, 0, 0, 1]
        # w: C11=1, C22=1 => [1, 0, 0, 1] (Note: C11 is index 0, C22 is index 3)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # m2 = (c+d)e
        # u: c=1, d=1 => [0, 0, 1, 1]
        # v: e=1 => [1, 0, 0, 0]
        # w: C21=1 => [0, 0, 1, 0]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        
        # m3 = a(f-h)
        # u: a=1 => [1, 0, 0, 0]
        # v: f=1, h=-1 => [0, 1, 0, -1]
        # w: C12=1, C22=1 => [0, 1, 0, 1]
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        
        # m4 = d(g-e)
        # u: d=1 => [0, 0, 0, 1]
        # v: g=1, e=-1 => [-1, 0, 1, 0]
        # w: C11=1, C21=1 => [1, 0, 1, 0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        
        # m5 = (a+b)h
        # u: a=1, b=1 => [1, 1, 0, 0]
        # v: h=1 => [0, 0, 0, 1]
        # w: C11=-1, C12=1 => [-1, 1, 0, 0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        
        # m6 = (c-d)(f+h)
        # u: c=1, d=-1 => [0, 0, 1, -1]
        # v: f=1, h=1 => [0, 1, 0, 1]
        # w: C22=1 => [0, 0, 0, 1]
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        
        # m7 = (b-d)(e+g)
        # u: b=1, d=-1 => [0, 1, 0, -1]
        # v: e=1, g=1 => [1, 0, 1, 0]
        # w: C11=1 => [1, 0, 0, 0]
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    
    col = 0
    for block_idx in range(num_blocks):
        for term in strassen_terms:
            u, v, w = term
            # Fill U
            for i in range(4):
                U[i][col] = u[i]
            # Fill V
            # B is 2x12. Block idx corresponds to columns 2*block_idx and 2*block_idx+1
            # v[0] is e (B[0, 2*block_idx])
            # v[1] is f (B[0, 2*block_idx+1])
            # v[2] is g (B[1, 2*block_idx])
            # v[3] is h (B[1, 2*block_idx+1])
            # V index is j*p + k. j is row of B (0 or 1), k is column of B (0..11)
            # v[0] -> j=0, k=2*block_idx -> index 0*12 + 2*block_idx
            # v[1] -> j=0, k=2*block_idx+1 -> index 0*12 + 2*block_idx+1
            # v[2] -> j=1, k=2*block_idx -> index 1*12 + 2*block_idx
            # v[3] -> j=1, k=2*block_idx+1 -> index 1*12 + 2*block_idx+1
            V[0 * p + 2 * block_idx][col] = v[0]
            V[0 * p + 2 * block_idx + 1][col] = v[1]
            V[1 * p + 2 * block_idx][col] = v[2]
            V[1 * p + 2 * block_idx + 1][col] = v[3]
            
            # Fill W
            # W index is k*n + i. k is column of C (0..11), i is row of C (0 or 1)
            # w[0] is C11 (i=0, k=2*block_idx) -> index 2*block_idx*2 + 0
            # w[1] is C12 (i=0, k=2*block_idx+1) -> index (2*block_idx+1)*2 + 0
            # w[2] is C21 (i=1, k=2*block_idx) -> index 2*block_idx*2 + 1
            # w[3] is C22 (i=1, k=2*block_idx+1) -> index (2*block_idx+1)*2 + 1
            W[2 * block_idx * n + 0][col] = w[0]
            W[(2 * block_idx + 1) * n + 0][col] = w[1]
            W[2 * block_idx * n + 1][col] = w[2]
            W[(2 * block_idx + 1) * n + 1][col] = w[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
