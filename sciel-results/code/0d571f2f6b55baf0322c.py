"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2 multiplication
    # u_s: 4 entries (i*m+j)
    # v_s: 8 entries (j*p+k), only k=0,1 used in base
    # w_s: 8 entries (k*n+i), only k=0,1 used in base
    
    # M0: (a00+a11)(b00+b11) -> C00+1, C11+1
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 0, 1, 0, 0, 0] # b00, b11
    w0 = [1, 0, 0, 1, 0, 0, 0, 0] # C00, C11
    
    # M1: (a10+a11)b00 -> C10+1, C11-1
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0, 0, 0, 0, 0] # b00
    w1 = [0, 1, 0, -1, 0, 0, 0, 0] # C10, C11
    
    # M2: a00(b01-b11) -> C01+1, C11+1
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, 0, 0, -1, 0, 0] # b01, -b11
    w2 = [0, 0, 1, 1, 0, 0, 0, 0] # C01, C11
    
    # M3: a11(b10-b00) -> C00+1, C10+1
    u3 = [0, 0, 0, 1]
    v3 = [-1, 0, 0, 0, 1, 0, 0, 0] # -b00, b10
    w3 = [1, 1, 0, 0, 0, 0, 0, 0] # C00, C10
    
    # M4: (a00+a01)b11 -> C00-1, C01+1
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 0, 0, 1, 0, 0] # b11
    w4 = [-1, 0, 1, 0, 0, 0, 0, 0] # C00, C01
    
    # M5: (a10-a00)(b00+b01) -> C11+1
    u5 = [-1, 0, 1, 0]
    v5 = [1, 1, 0, 0, 0, 0, 0, 0] # b00, b01
    w5 = [0, 0, 0, 1, 0, 0, 0, 0] # C11
    
    # M6: (a01-a11)(b10+b11) -> C00+1
    u6 = [0, 1, 0, -1]
    v6 = [0, 0, 0, 0, 1, 1, 0, 0] # b10, b11
    w6 = [1, 0, 0, 0, 0, 0, 0, 0] # C00
    
    base_u = [u0, u1, u2, u3, u4, u5, u6]
    base_v = [v0, v1, v2, v3, v4, v5, v6]
    base_w = [w0, w1, w2, w3, w4, w5, w6]
    
    # Fill U, V, W for first block (k=0,1)
    for s in range(7):
        r = s
        for idx in range(n * m):
            U[idx][r] = base_u[s][idx]
        for idx in range(m * p):
            V[idx][r] = base_v[s][idx]
        for idx in range(p * n):
            W[idx][r] = base_w[s][idx]
            
    # Fill U, V, W for second block (k=2,3)
    for s in range(7):
        r = 7 + s
        # U is same
        for idx in range(n * m):
            U[idx][r] = base_u[s][idx]
        
        # V is shifted: k' -> k'+2
        for j in range(m):
            for k in range(2):
                val = base_v[s][j * p + k]
                V[j * p + (k + 2)][r] = val
        
        # W is shifted: k' -> k'+2
        # Index in W is k*n + i
        # Old index k'*n + i, new index (k'+2)*n + i = old + 2*n
        for idx in range(p * n):
            if idx < 2 * n:
                W[idx + 2 * n][r] = base_w[s][idx]
            # Entries for k=0,1 in base_w are at indices 0..2n-1
            # They map to 2n..3n-1
            # Entries for k=2,3 in base_w are 0, so they map to 4n..5n-1 (which is out of bounds or zero)
            # Wait, base_w has length 8. Indices 0..7.
            # k=0,1 correspond to indices 0..3.
            # k=2,3 correspond to indices 4..7 (which are 0 in base_w).
            # So we only need to shift indices 0..3 to 4..7.
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
