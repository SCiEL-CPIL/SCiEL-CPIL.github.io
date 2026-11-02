import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x10 multiplication into 5 blocks of 2x2 multiplication.
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # Total rank = 5 * 7 = 35.
    
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # W indices: 00, 10, 01, 11 -> 0, 1, 2, 3 (local, k*n+i)
    
    # Term 1: M1 = (a+d)(e+h)
    U1 = [1, 0, 1, 0]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, 1] # C00, C11
    
    # Term 2: M2 = (c+d)e
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, 1] # C10, C11
    
    # Term 3: M3 = a(f-h)
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, -1] # C01, C11
    
    # Term 4: M4 = d(g-e)
    U4 = [0, 0, 0, 1]
    V4 = [-1, 0, 1, 0]
    W4 = [-1, 0, 1, 0] # C00, C01
    
    # Term 5: M5 = (a+b)h
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [1, 1, 0, 0] # C00, C10
    
    # Term 6: M6 = (c-a)(e+f)
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [1, 0, 0, 0] # C00
    
    # Term 7: M7 = (b-d)(g+h)
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [0, 1, 0, -1] # C10, C11
    
    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    col = 0
    for l in range(num_blocks):
        for r in range(rank_per_block):
            u_vec, v_vec, w_vec = strassen_terms[r]
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u_vec[i * m + j]
            
            # Fill V
            # Local V index j, k (k in 0..1) maps to global j*p + (2*l + k)
            for j in range(m):
                for k in range(2):
                    local_idx = j * 2 + k
                    global_idx = j * p + (2 * l + k)
                    V[global_idx][col] = v_vec[local_idx]
            
            # Fill W
            # Local W index k, i (k in 0..1) maps to global (2*l + k)*n + i
            for k in range(2):
                for i in range(n):
                    local_idx = k * n + i
                    global_idx = (2 * l + k) * n + i
                    W[global_idx][col] = w_vec[local_idx]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
