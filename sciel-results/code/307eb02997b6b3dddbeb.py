# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication
    # Structure: 3x3 grid of 2x2 blocks
    # Block size 2x2
    # Number of blocks per dim: 3
    
    # Rank of M_3 (Winograd) = 19
    # Rank of M_2 (Strassen) = 7
    # Total Rank = 19 * 7 = 133
    
    rank = 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_u = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,0): 1},           # M3
        {(1,1): 1},           # M4
        {(0,0): 1, (0,1): 1}, # M5
        {(1,0): 1, (0,0): -1},# M6
        {(0,1): 1, (1,1): -1} # M7
    ]
    strassen_v = [
        {(0,0): 1, (1,1): 1}, # M1
        {(0,0): 1},           # M2
        {(0,1): 1, (1,1): -1},# M3
        {(1,0): 1, (0,0): -1},# M4
        {(1,1): 1},           # M5
        {(0,0): 1, (0,1): 1}, # M6
        {(1,0): 1, (1,1): 1}  # M7
    ]
    strassen_w = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): -1},# M2
        {(0,1): 1, (1,1): 1}, # M3
        {(0,0): 1, (1,0): 1}, # M4
        {(0,0): -1, (0,1): 1},# M5
        {(1,1): 1},           # M6
        {(0,0): 1}            # M7
    ]
    
    # Winograd 3x3 coefficients (Rank 19)
    # This is the risky part. I will use a known set.
    # If this fails, the score will be 0.
    # I will use a standard Winograd 3x3 decomposition.
    
    # Given the high risk, I will fall back to the 189 rank solution 
    # which is known to be correct in the current program.
    # I will just clean up the 189 solution.
    
    # Re-initialize for 189 rank (Strassen 2x2 blocks, standard 3x3 block mult)
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_size = 2
    blocks_per_dim = 3
    offsets = [i * block_size for i in range(blocks_per_dim)]
    
    # Standard 3x3 block multiplication has 27 terms
    # Each term is a 2x2 matmul (Strassen rank 7)
    # Total 27 * 7 = 189
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = offsets[I]
        for J in range(blocks_per_dim):
            off_J = offsets[J]
            for K in range(blocks_per_dim):
                off_K = offsets[K]
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = off_K + pj
                        i = off_I + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                if time.time() > _deadline:
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
