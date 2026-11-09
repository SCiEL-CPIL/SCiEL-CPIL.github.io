"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 independent Strassen (2x2x2) algorithms
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # Each term is a list of (index, coeff) for U, V, W
    # U indices: (i, j) -> i*2 + j
    # V indices: (j, k) -> j*2 + k
    # W indices: (k, i) -> k*2 + i
    
    strassen_terms = [
        # Term 1: (a+d)(e+h)
        {
            'U': [(0, 1), (3, 1)],  # a, d
            'V': [(0, 1), (3, 1)],  # e, h
            'W': [(0, 1), (3, 1)]   # C00, C11
        },
        # Term 2: (c+d)e
        {
            'U': [(2, 1), (3, 1)],  # c, d
            'V': [(0, 1)],          # e
            'W': [(1, 1), (3, -1)]  # C10 (k=0,i=1), C11 (k=1,i=1)
        },
        # Term 3: a(f-h)
        {
            'U': [(0, 1)],          # a
            'V': [(1, 1), (3, -1)], # f, h
            'W': [(2, 1), (3, 1)]   # C01 (k=1,i=0), C11 (k=1,i=1)
        },
        # Term 4: d(g-e)
        {
            'U': [(3, 1)],          # d
            'V': [(2, 1), (0, -1)], # g, e
            'W': [(0, 1), (1, 1)]   # C00 (k=0,i=0), C10 (k=0,i=1)
        },
        # Term 5: (a+b)h
        {
            'U': [(0, 1), (1, 1)],  # a, b
            'V': [(3, 1)],          # h
            'W': [(0, -1), (2, 1)]  # C00 (k=0,i=0), C01 (k=1,i=0)
        },
        # Term 6: (c-a)(e+f)
        {
            'U': [(2, 1), (0, -1)], # c, a
            'V': [(0, 1), (1, 1)],  # e, f
            'W': [(3, 1)]           # C11 (k=1,i=1)
        },
        # Term 7: (b-d)(g+h)
        {
            'U': [(1, 1), (3, -1)], # b, d
            'V': [(2, 1), (3, 1)],  # g, h
            'W': [(0, 1)]           # C00 (k=0,i=0)
        }
    ]
    
    # Define the 4 blocks
    # Block 0: A_L (j=0,1) * B_LL (k=0,1)
    # Block 1: A_L (j=0,1) * B_LR (k=2,3)
    # Block 2: A_R (j=2,3) * B_RL (k=0,1)
    # Block 3: A_R (j=2,3) * B_RR (k=2,3)
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            # Add to U
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
