"""Matrix multiplication 2x2 x 2x10 rank-minimal decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # Use Strassen's algorithm applied to 5 blocks of 2 columns each.
    # For each block of 2 columns, we use 7 multiplications instead of 4.
    # Total rank = 5 * 7 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = A[0,0] * (B[0,0] + B[1,0])
    # M2 = (A[0,0] + A[0,1]) * B[1,0]
    # M3 = (A[1,0] + A[1,1]) * B[0,0]
    # M4 = A[1,1] * (B[0,1] - B[1,1])
    # M5 = (A[0,0] + A[1,1]) * (B[1,1] - B[0,0])
    # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # M7 = (A[0,0] - A[1,0]) * (B[0,0] + B[0,1])
    
    # For a block of 2 columns (columns 2b and 2b+1):
    # B[0, 2b], B[1, 2b], B[0, 2b+1], B[1, 2b+1]
    
    for b in range(5):
        base = 7 * b
        # Column indices for this block
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = A[0,0] * (B[0,k0] + B[1,k0])
        r = base + 0
        U[0][r] = 1  # A[0,0]
        V[k0][r] = 1  # B[0,k0]
        V[10 + k0][r] = 1  # B[1,k0]
        # Contributes to C[0,k0]: W[k0*2+0][r] = 1
        W[k0 * 2 + 0][r] = 1
        # Contributes to C[1,k0]: W[k0*2+1][r] = 0 (handled by other terms)
        
        # M2 = (A[0,0] + A[0,1]) * B[1,k0]
        r = base + 1
        U[0][r] = 1  # A[0,0]
        U[1][r] = 1  # A[0,1]
        V[10 + k0][r] = 1  # B[1,k0]
        # Contributes to C[0,k0]: W[k0*2+0][r] = 1
        W[k0 * 2 + 0][r] = 1
        # Contributes to C[0,k1]: W[k1*2+0][r] = 0
        # Contributes to C[1,k0]: W[k0*2+1][r] = 0
        
        # M3 = (A[1,0] + A[1,1]) * B[0,k0]
        r = base + 2
        U[2][r] = 1  # A[1,0]
        U[3][r] = 1  # A[1,1]
        V[k0][r] = 1  # B[0,k0]
        # Contributes to C[1,k0]: W[k0*2+1][r] = 1
        W[k0 * 2 + 1][r] = 1
        
        # M4 = A[1,1] * (B[0,k1] - B[1,k1])
        r = base + 3
        U[3][r] = 1  # A[1,1]
        V[k1][r] = 1  # B[0,k1]
        V[10 + k1][r] = -1  # B[1,k1]
        # Contributes to C[1,k1]: W[k1*2+1][r] = 1
        W[k1 * 2 + 1][r] = 1
        
        # M5 = (A[0,0] + A[1,1]) * (B[1,k1] - B[0,k0])
        r = base + 4
        U[0][r] = 1  # A[0,0]
        U[3][r] = 1  # A[1,1]
        V[10 + k1][r] = 1  # B[1,k1]
        V[k0][r] = -1  # B[0,k0]
        # Contributes to C[0,k0]: W[k0*2+0][r] = 1
        W[k0 * 2 + 0][r] = 1
        # Contributes to C[0,k1]: W[k1*2+0][r] = 1
        W[k1 * 2 + 0][r] = 1
        # Contributes to C[1,k0]: W[k0*2+1][r] = 1
        W[k0 * 2 + 1][r] = 1
        # Contributes to C[1,k1]: W[k1*2+1][r] = 1
        W[k1 * 2 + 1][r] = 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        r = base + 5
        U[1][r] = 1  # A[0,1]
        U[3][r] = -1  # A[1,1]
        V[10 + k0][r] = 1  # B[1,k0]
        V[10 + k1][r] = 1  # B[1,k1]
        # Contributes to C[0,k0]: W[k0*2+0][r] = 1
        W[k0 * 2 + 0][r] = 1
        # Contributes to C[0,k1]: W[k1*2+0][r] = 1
        W[k1 * 2 + 0][r] = 1
        
        # M7 = (A[0,0] - A[1,0]) * (B[0,k0] + B[0,k1])
        r = base + 6
        U[0][r] = 1  # A[0,0]
        U[2][r] = -1  # A[1,0]
        V[k0][r] = 1  # B[0,k0]
        V[k1][r] = 1  # B[0,k1]
        # Contributes to C[1,k0]: W[k0*2+1][r] = 1
        W[k0 * 2 + 1][r] = 1
        # Contributes to C[1,k1]: W[k1*2+1][r] = 1
        W[k1 * 2 + 1][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
