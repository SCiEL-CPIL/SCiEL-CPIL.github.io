"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Define Strassen blocks
    # Each block is (I, J, K) where I, J, K are lists of 2 indices
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [4, 5])
    ]
    
    # Calculate rank
    rank_per_block = 7
    num_blocks = len(blocks)
    covered_triples = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered_triples.add((i, j, k))
                    
    total_triples = n * m * p
    num_trivial = total_triples - len(covered_triples)
    total_rank = num_blocks * rank_per_block + num_trivial
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # U_coefs[term][local_i][local_j]
    # V_coefs[term][local_j][local_k]
    # W_coefs[term][local_k][local_i]
    
    # M1
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, -1]]
    
    # M2
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 0], [1, 0]]
    
    # M3
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 0], [1, 0]]
    W3 = [[0, 0], [0, 1]]
    
    # M4
    U4 = [[0, 0], [0, 1]]
    V4 = [[0, 0], [0, 1]]
    W4 = [[1, 0], [1, 1]]
    
    # M5
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 0], [1, 0]]
    
    # M6
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [1, 0]]
    
    # M7
    U7 = [[0, 1], [-1, 0]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [0, -1]]
    
    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    for I, J, K in blocks:
        for term_idx in range(7):
            Uc, Vc, Wc = strassen_terms[term_idx]
            for li in range(2):
                for lj in range(2):
                    val_u = Uc[li][lj]
                    if val_u != 0:
                        i = I[li]
                        j = J[lj]
                        U[i * m + j][col] = val_u
                        
            for lj in range(2):
                for lk in range(2):
                    val_v = Vc[lj][lk]
                    if val_v != 0:
                        j = J[lj]
                        k = K[lk]
                        V[j * p + k][col] = val_v
                        
            for lk in range(2):
                for li in range(2):
                    val_w = Wc[lk][li]
                    if val_w != 0:
                        k = K[lk]
                        i = I[li]
                        W[k * n + i][col] = val_w
                        
            col += 1
            
    # Trivial terms for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_triples:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
