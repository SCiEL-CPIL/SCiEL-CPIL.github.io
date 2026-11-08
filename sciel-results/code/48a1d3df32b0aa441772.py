"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m + j, c] for A[i,j]
    # V[j*p + k, c] for B[j,k]
    # W[k*n + i, c] for C[i,k]  (C is n x p)
    
    # We'll build the decomposition using Strassen for 2x2 by 2x2 blocks.
    # Structure: A (2x4) = [A_L | A_R], B (4x8) = [B_top; B_bot]
    # C = A_L * B_top + A_R * B_bot
    # A_L*B_top: A_L is 2x2 at (i0=0,j0=0), B_top is 2x8 at rows 0-1
    #   Split B_top into 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7
    # A_R*B_bot: A_R is 2x2 at (i0=0,j0=2), B_bot is 2x8 at rows 2-3
    #   Split B_bot into 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7
    # Total: 8 Strassen blocks, each rank 7 => rank 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2 by 2x2 block.
        A sub-block: rows i0..i0+1, cols j0..j0+1
        B sub-block: rows j0..j0+1, cols k0..k0+1
        C sub-block: rows k0..k0+1, cols i0..i0+1
        Returns new column index.
        """
        nonlocal col
        # A entries: a11=A[i0,j0], a12=A[i0,j0+1], a21=A[i0+1,j0], a22=A[i0+1,j0+1]
        a11_idx = i0 * m + j0
        a12_idx = i0 * m + (j0 + 1)
        a21_idx = (i0 + 1) * m + j0
        a22_idx = (i0 + 1) * m + (j0 + 1)
        
        # B entries: b11=B[j0,k0], b12=B[j0,k0+1], b21=B[j0+1,k0], b22=B[j0+1,k0+1]
        b11_idx = j0 * p + k0
        b12_idx = j0 * p + (k0 + 1)
        b21_idx = (j0 + 1) * p + k0
        b22_idx = (j0 + 1) * p + (k0 + 1)
        
        # C entries: c11=C[i0,k0], c12=C[i0,k0+1], c21=C[i0+1,k0], c22=C[i0+1,k0+1]
        # W index for C[i,k] is k*n + i
        c11_idx = k0 * n + i0
        c12_idx = (k0 + 1) * n + i0
        c21_idx = k0 * n + (i0 + 1)
        c22_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11+a22)*(b11+b22)
        # C11 += M1, C22 -= M1
        U[a11_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # M2 = (a21+a22)*b11
        # C21 += M2, C22 -= M2
        U[a21_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        W[c21_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # M3 = a11*(b12-b22)
        # C12 += M3, C22 += M3
        U[a11_idx][col] = 1
        V[b12_idx][col] = 1
        V[b22_idx][col] = -1
        W[c12_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # M4 = a22*(b21-b11)
        # C11 += M4, C21 += M4
        U[a22_idx][col] = 1
        V[b21_idx][col] = 1
        V[b11_idx][col] = -1
        W[c11_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # M5 = (a11+a12)*b22
        # C12 += M5, C11 -= M5
        U[a11_idx][col] = 1
        U[a12_idx][col] = 1
        V[b22_idx][col] = 1
        W[c12_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # M6 = (a21-a11)*(b11+b12)
        # C22 += M6
        U[a21_idx][col] = 1
        U[a11_idx][col] = -1
        V[b11_idx][col] = 1
        V[b12_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # M7 = (a12-a22)*(b21+b22)
        # C11 += M7
        U[a12_idx][col] = 1
        U[a22_idx][col] = -1
        V[b21_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        return col
    
    # A_L * B_top: A_L at (i0=0, j0=0), B_top rows 0-1
    for k0 in [0, 2, 4, 6]:
        add_strassen_block(i0=0, j0=0, k0=k0)
    
    # A_R * B_bot: A_R at (i0=0, j0=2), B_bot rows 2-3
    for k0 in [0, 2, 4, 6]:
        add_strassen_block(i0=0, j0=2, k0=k0)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
