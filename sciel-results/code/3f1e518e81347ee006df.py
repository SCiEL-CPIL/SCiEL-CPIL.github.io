"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank is 6 blocks * 7 Strassen terms = 42
    rank = 42
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 matrix multiplication
    # u_s corresponds to A
    # v_s corresponds to B
    # w_s corresponds to C
    # Indices:
    # A: [a00, a01, a10, a11]
    # B: [b00, b01, b10, b11]
    # C: [c00, c01, c10, c11]
    
    u_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v_vecs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    r = 0
    for b in range(6):
        for s in range(7):
            # Fill U
            for i in range(4):
                U[i][r] = u_vecs[s][i]
            
            # Fill V
            # v_vecs[s] is [v00, v01, v10, v11]
            # j is 0 or 1
            # k' is 0 or 1
            # Global k = 2*b + k'
            # Global index in V: j * p + k
            for j in range(2):
                for kp in range(2):
                    local_val = v_vecs[s][j * 2 + kp]
                    global_k = 2 * b + kp
                    global_idx = j * p + global_k
                    V[global_idx][r] = local_val
            
            # Fill W
            # w_vecs[s] is [w00, w01, w10, w11] where indices are (k', i)
            # Wait, w_vecs are defined for C.
            # C indices: c00, c01, c10, c11.
            # c_ij corresponds to i (row) and j (col).
            # In our tensor, W index is k*n + i.
            # Here k is the column of B, which is also the column of C.
            # i is the row of A, which is also the row of C.
            # So C_{i, k} corresponds to W[k*n + i].
            # My w_vecs are indexed as [c00, c01, c10, c11].
            # c00: i=0, k'=0.
            # c01: i=0, k'=1.
            # c10: i=1, k'=0.
            # c11: i=1, k'=1.
            # So w_vecs[s][idx] where idx = i*2 + k'?
            # Let's check my w_vecs definition.
            # w1 = [1, 0, 0, 1] -> c00=1, c01=0, c10=0, c11=1.
            # This matches c00 and c11 being 1.
            # So index 0 is i=0, k'=0.
            # Index 1 is i=0, k'=1.
            # Index 2 is i=1, k'=0.
            # Index 3 is i=1, k'=1.
            # So w_vecs[s][i*2 + k'].
            
            for i in range(2):
                for kp in range(2):
                    local_val = w_vecs[s][i * 2 + kp]
                    global_k = 2 * b + kp
                    global_idx = global_k * n + i
                    W[global_idx][r] = local_val
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
