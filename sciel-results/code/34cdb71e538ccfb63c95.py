"""Matrix multiplication 2x2 x 2x10 rank-minimal decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # Use Strassen's 7-multiplication algorithm for 2x2 * 2x2 blocks.
    # Split the 10 columns into 5 blocks of 2 columns each.
    # For each block, we use 7 multiplications instead of 4*2=8.
    # Total rank = 5 * 7 = 35.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = a*(e+h)
    # M2 = (c+d)*e
    # M3 = (a+b)*h
    # M4 = d*(f-g)
    # M5 = (a+c)*(e+f)
    # M6 = (b-d)*(g+h)
    # M7 = (a-c)*(e-g)
    # 
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,k]=k, B[1,k]=10+k for k=0..9
    # C indices: C[i,k] -> W index = k*n + i = k*2 + i
    
    for block in range(5):
        # Columns for this block: k0 = 2*block, k1 = 2*block + 1
        k0 = 2 * block
        k1 = 2 * block + 1
        base = block * 7
        
        # M1 = A[0,0] * (B[0,k0] + B[1,k0])
        U[0][base + 0] = 1
        V[k0][base + 0] = 1
        V[10 + k0][base + 0] = 1
        # C[0,k0] += M1, C[1,k1] += M1
        W[k0 * 2 + 0][base + 0] = 1
        W[k1 * 2 + 1][base + 0] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k0][base + 1] = 1
        # C[1,k0] += M2, C[1,k1] -= M2
        W[k0 * 2 + 1][base + 1] = 1
        W[k1 * 2 + 1][base + 1] = -1
        
        # M3 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0][base + 2] = 1
        U[1][base + 2] = 1
        V[10 + k1][base + 2] = 1
        # C[0,k1] += M3, C[1,k1] += M3
        W[k1 * 2 + 0][base + 2] = 1
        W[k1 * 2 + 1][base + 2] = 1
        
        # M4 = A[1,1] * (B[0,k1] - B[1,k1])
        U[3][base + 3] = 1
        V[k1][base + 3] = 1
        V[10 + k1][base + 3] = -1
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0 * 2 + 0][base + 3] = 1
        W[k0 * 2 + 1][base + 3] = 1
        
        # M5 = (A[0,0] + A[1,0]) * (B[0,k0] + B[1,k0])
        U[0][base + 4] = 1
        U[2][base + 4] = 1
        V[k0][base + 4] = 1
        V[10 + k0][base + 4] = 1
        # C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * 2 + 0][base + 4] = -1
        W[k1 * 2 + 0][base + 4] = 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1][base + 5] = 1
        U[3][base + 5] = -1
        V[10 + k0][base + 5] = 1
        V[10 + k1][base + 5] = 1
        # C[1,k1] += M6
        W[k1 * 2 + 1][base + 5] = 1
        
        # M7 = (A[0,0] - A[1,0]) * (B[0,k0] - B[1,k0])
        U[0][base + 6] = 1
        U[2][base + 6] = -1
        V[k0][base + 6] = 1
        V[10 + k0][base + 6] = -1
        # C[0,k0] += M7
        W[k0 * 2 + 0][base + 6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
