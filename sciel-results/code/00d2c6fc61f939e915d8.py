"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    Strategy:
    - Split the multiplication into four 2x2x2 blocks corresponding to columns (0,1) of B for rows 0 and 1.
    - Use Strassen's algorithm for each 2x2x2 block (rank 7 instead of 8).
    - Treat the third column of B (index 2) separately using schoolbook multiplication (rank 16).
    - Total rank: 4 * 7 + 16 = 44.
    
    Interface: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    U rows: i*m + j for A[i,j]
    V rows: j*p + k for B[j,k]
    W rows: k*n + i for C[i,k] (Trace convention W[k*n+i, r])
    """
    n, m, p = DIMENSIONS
    total_rank = 44
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # A_block is A[0:2, 0:2]
    # B_block is B[0:2, 2*r : 2*r+2]
    # C_block is C[0:2, 2*r : 2*r+2]
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M6
    
    # Indices in global U, V, W
    # U indices for A[0,0], A[0,1], A[1,0], A[1,1] are 0, 1, 3, 4
    # V indices for B[0, 2r], B[0, 2r+1], B[1, 2r], B[1, 2r+1]
    #   B[0, 2r] -> 0*8 + 2r = 2r
    #   B[0, 2r+1] -> 0*8 + 2r+1 = 2r+1
    #   B[1, 2r] -> 1*8 + 2r = 8+2r
    #   B[1, 2r+1] -> 1*8 + 2r+1 = 8+2r+1
    # W indices for C[0, 2r], C[0, 2r+1], C[1, 2r], C[1, 2r+1]
    #   C[0, 2r] -> 2r*2 + 0 = 4r
    #   C[0, 2r+1] -> (2r+1)*2 + 0 = 4r+2
    #   C[1, 2r] -> 2r*2 + 1 = 4r+1
    #   C[1, 2r+1] -> (2r+1)*2 + 1 = 4r+3
    
    for r in range(4):
        # V indices
        v00 = 2*r
        v01 = 2*r + 1
        v10 = 8 + 2*r
        v11 = 8 + 2*r + 1
        
        # W indices
        w00 = 4*r
        w01 = 4*r + 2
        w10 = 4*r + 1
        w11 = 4*r + 3
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] = 1
        U[4][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[3][col] = 1
        U[4][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0][col] = 1
        V[v10][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[4][col] = 1
        V[v01][col] = 1
        V[v00][col] = -1
        W[w00][col] = -1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] = 1
        U[1][col] = 1
        V[v11][col] = 1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[3][col] = 1
        U[0][col] = -1
        V[v00][col] = 1
        V[v10][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        U[1][col] = 1
        U[4][col] = -1
        V[v01][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1

    # Handle the third column of B (index 2) with schoolbook
    # A[0,2] is U index 2
    # A[1,2] is U index 5
    # B[2, k] is V index 16 + k
    # C[0, k] is W index 2k
    # C[1, k] is W index 2k + 1
    
    for k in range(8):
        # Term: A[0,2] * B[2, k] -> C[0, k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # Term: A[1,2] * B[2, k] -> C[1, k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    # col should be 44
    assert col == total_rank, f"Rank mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
