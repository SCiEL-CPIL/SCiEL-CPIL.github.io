"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_1 B_1 (2x1x8)
    # A_1 is col 0 of A (indices 0, 3)
    # B_1 is row 0 of B (indices 0..7)
    # C indices: k*n + i
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A_2 B_2 (2x2x8) using 4 Strassen blocks
    # A_2 is cols 1,2 of A
    # B_2 is rows 1,2 of B
    # Strassen for 2x2x2
    # A_2 indices in U: i*m + j
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=1, j=1 -> 4
    # i=1, j=2 -> 5
    
    # B_2 indices in V: j*p + k
    # j=1, k=2b -> 1*8 + 2b = 8+2b
    # j=1, k=2b+1 -> 8+2b+1
    # j=2, k=2b -> 2*8 + 2b = 16+2b
    # j=2, k=2b+1 -> 16+2b+1
    
    # C indices in W: k*n + i
    # C_0, 2b -> (2b)*2 + 0 = 4b
    # C_0, 2b+1 -> (2b+1)*2 + 0 = 4b+2
    # C_1, 2b -> (2b)*2 + 1 = 4b+1
    # C_1, 2b+1 -> (2b+1)*2 + 1 = 4b+3

    # Strassen terms for one 2x2x2 block
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2: (A10+A11)B00 -> C10+=1
    # M3: A00(B01-B11) -> C01+=1
    # M4: A11(B10-B00) -> C10+=1, C11+=-1
    # M5: (A00+A01)B11 -> C00+=-1, C01+=1
    # M6: (A10-A00)(B00+B01) -> C11+=1
    # M7: (A01-A11)(B10+B11) -> C00+=1, C10+=-1

    # Note: A00=A_2[0,0]=A[0,1], A01=A_2[0,1]=A[0,2]
    # A10=A_2[1,0]=A[1,1], A11=A_2[1,1]=A[1,2]
    # B00=B_2[0,0]=B[1,0], B01=B_2[0,1]=B[1,1]
    # B10=B_2[1,0]=B[2,0], B11=B_2[1,1]=B[2,1]

    # We will iterate over 4 blocks b=0..3
    for b in range(4):
        # Local indices for A_2
        a00_idx = 0 * m + 1
        a01_idx = 0 * m + 2
        a10_idx = 1 * m + 1
        a11_idx = 1 * m + 2
        
        # Local indices for B_2 block b
        k0 = 2 * b
        k1 = 2 * b + 1
        b00_idx = 1 * p + k0
        b01_idx = 1 * p + k1
        b10_idx = 2 * p + k0
        b11_idx = 2 * p + k1
        
        # Local indices for C block b
        c00_idx = k0 * n + 0
        c01_idx = k1 * n + 0
        c10_idx = k0 * n + 1
        c11_idx = k1 * n + 1
        
        # M1
        U[a00_idx][r] = 1
        U[a11_idx][r] = 1
        V[b00_idx][r] = 1
        V[b11_idx][r] = 1
        W[c00_idx][r] = 1
        W[c11_idx][r] = 1
        r += 1
        
        # M2
        U[a10_idx][r] = 1
        U[a11_idx][r] = 1
        V[b00_idx][r] = 1
        W[c10_idx][r] = 1
        r += 1
        
        # M3
        U[a00_idx][r] = 1
        V[b01_idx][r] = 1
        V[b11_idx][r] = -1
        W[c01_idx][r] = 1
        r += 1
        
        # M4
        U[a11_idx][r] = 1
        V[b10_idx][r] = 1
        V[b00_idx][r] = -1
        W[c10_idx][r] = 1
        W[c11_idx][r] = -1
        r += 1
        
        # M5
        U[a00_idx][r] = 1
        U[a01_idx][r] = 1
        V[b11_idx][r] = 1
        W[c00_idx][r] = -1
        W[c01_idx][r] = 1
        r += 1
        
        # M6
        U[a10_idx][r] = 1
        U[a00_idx][r] = -1
        V[b00_idx][r] = 1
        V[b01_idx][r] = 1
        W[c11_idx][r] = 1
        r += 1
        
        # M7
        U[a01_idx][r] = 1
        U[a11_idx][r] = -1
        V[b10_idx][r] = 1
        V[b11_idx][r] = 1
        W[c00_idx][r] = 1
        W[c10_idx][r] = -1
        r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
