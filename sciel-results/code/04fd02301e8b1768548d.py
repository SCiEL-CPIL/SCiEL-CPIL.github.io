import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a block-strassen decomposition with rank 28 (4 blocks of rank 7).
    """
    n, m, p = DIMENSIONS
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 multiplication
    # u: [a, b, c, d]
    # v: [e, f, g, h]
    # w: [p, q, r, s]
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, -1, 1, 0],# M6
        [0, 1, -1, 0] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1 -> p, s
        [0, 0, 1, 0], # M2 -> r
        [0, 1, 0, 1], # M3 -> q, s
        [1, 0, 1, 0], # M4 -> p, r
        [-1, 1, 0, 0],# M5 -> -p, q
        [0, 0, 0, 1], # M6 -> s
        [1, 0, 0, 0]  # M7 -> p
    ]
    
    # Define the 4 blocks
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        {'a_j': 0, 'b_j': 0, 'c_k': 0}, # Block 0
        {'a_j': 2, 'b_j': 2, 'c_k': 0}, # Block 1
        {'a_j': 0, 'b_j': 0, 'c_k': 2}, # Block 2
        {'a_j': 2, 'b_j': 2, 'c_k': 2}  # Block 3
    ]
    
    r = 0
    for block in blocks:
        a_j_off = block['a_j']
        b_j_off = block['b_j']
        c_k_off = block['c_k']
        
        for t in range(7):
            u_vec = strassen_u[t]
            v_vec = strassen_v[t]
            w_vec = strassen_w[t]
            
            # Fill U
            # Local indices (li, lj) map to global (i, j)
            # i is 0, 1 (since n=2 and we don't split i)
            # j is a_j_off + lj
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = a_j_off + lj
                    u_val = u_vec[li * 2 + lj]
                    if u_val != 0:
                        U[gi * m + gj, r] = u_val
            
            # Fill V
            # Local indices (lj, lk) map to global (j, k)
            # j is b_j_off + lj
            # k is c_k_off + lk
            for lj in range(2):
                for lk in range(2):
                    gj = b_j_off + lj
                    gk = c_k_off + lk
                    v_val = v_vec[lj * 2 + lk]
                    if v_val != 0:
                        V[gj * p + gk, r] = v_val
            
            # Fill W
            # Local indices (li, lk) map to global (i, k)
            # i is li
            # k is c_k_off + lk
            # W index is k * n + i
            for li in range(2):
                for lk in range(2):
                    gi = li
                    gk = c_k_off + lk
                    w_val = w_vec[li * 2 + lk]
                    if w_val != 0:
                        W[gk * n + gi, r] = w_val
            
            r += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
