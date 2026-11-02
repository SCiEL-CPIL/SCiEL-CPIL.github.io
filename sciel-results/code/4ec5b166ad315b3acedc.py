"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    def add_strassen_block(ai, aj, bi, bj, col_start):
        """Add 7 Strassen rank-1 tensors for A[ai:ai+2, aj:aj+2] * B[bi:bi+2, bj:bj+2]"""
        # ai, aj: top-left (i,j) of A block
        # bi, bj: top-left (j,k) of B block
        # A indices: (ai, aj), (ai, aj+1), (ai+1, aj), (ai+1, aj+1)
        # B indices: (bi, bj), (bi, bj+1), (bi+1, bj), (bi+1, bj+1)
        # Output C indices: (ai, bj), (ai, bj+1), (ai+1, bj), (ai+1, bj+1)
        
        a00 = ai * m + aj
        a01 = ai * m + (aj + 1)
        a10 = (ai + 1) * m + aj
        a11 = (ai + 1) * m + (aj + 1)
        
        b00 = bi * p + bj
        b01 = bi * p + (bj + 1)
        b10 = (bi + 1) * p + bj
        b11 = (bi + 1) * p + (bj + 1)
        
        c00 = bj * n + ai
        c01 = (bj + 1) * n + ai
        c10 = bj * n + (ai + 1)
        c11 = (bj + 1) * n + (ai + 1)
        
        # M1 = (a00+a11)*(b00+b11) -> C[ai,bj] +1, C[ai+1,bj+1] +1
        c = col_start + 0
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        
        # M2 = (a10+a11)*b00 -> C[ai+1,bj] +1, C[ai+1,bj+1] -1
        c = col_start + 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        
        # M3 = a00*(b10-b11) -> C[ai,bj+1] +1, C[ai+1,bj+1] +1
        c = col_start + 2
        U[a00][c] = 1
        V[b10][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        
        # M4 = a11*(b01-b00) -> C[ai,bj] +1, C[ai+1,bj] +1
        c = col_start + 3
        U[a11][c] = 1
        V[b01][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        
        # M5 = (a00+a01)*b11 -> C[ai,bj] -1, C[ai,bj+1] +1
        c = col_start + 4
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        
        # M6 = (a10-a00)*(b00+b10) -> C[ai+1,bj+1] +1
        c = col_start + 5
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b10][c] = 1
        W[c11][c] = 1
        
        # M7 = (a01-a11)*(b01+b11) -> C[ai,bj] +1, C[ai,bj+1] -1
        c = col_start + 6
        U[a01][c] = 1; U[a11][c] = -1
        V[b01][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c01][c] = -1

    # P1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0)
    
    # P2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 7)
    
    # P3: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 2, 0, 14)
    
    # P4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 2, 21)
    
    # Bottom row: A[2:3, 0:4] * B[0:4, 0:4] - schoolbook (16 multiplications)
    col = 28
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1  # A[2, j]
            V[j * p + k][col] = 1  # B[j, k]
            W[k * n + 2][col] = 1  # C[2, k]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
