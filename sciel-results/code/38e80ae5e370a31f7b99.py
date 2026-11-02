"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Strassen for 2x2 blocks
    # 6 blocks of 2x2 in the 2x12 matrix
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix indices i, j)
    # i in 0..1, j in 0..1
    # U[i*m + j, g]
    # c=0: M1 = (A00+A11)
    # c=1: M2 = (A10+A11)
    # c=2: M3 = A00
    # c=3: M4 = A11
    # c=4: M5 = (A00+A01)
    # c=5: M6 = (A10-A00)
    # c=6: M7 = (A01-A11)
    
    U_coeffs = [
        {0: 1, 3: 1},      # c=0: A00, A11
        {2: 1, 3: 1},      # c=1: A10, A11
        {0: 1},            # c=2: A00
        {3: 1},            # c=3: A11
        {0: 1, 1: 1},      # c=4: A00, A01
        {0: -1, 2: 1},     # c=5: -A00, A10
        {1: 1, 3: -1}      # c=6: A01, -A11
    ]
    
    # Strassen coefficients for V (B matrix indices j, k)
    # j in 0..1, k in 0..11
    # V[j*p + k, g]
    # For block b, k is 2b or 2b+1
    # c=0: M1 = (B00+B11) -> V[0, 2b]=1, V[1, 2b+1]=1
    # c=1: M2 = B00 -> V[0, 2b]=1
    # c=2: M3 = (B01-B11) -> V[0, 2b+1]=1, V[1, 2b+1]=-1
    # c=3: M4 = (B10-B00) -> V[1, 2b]=1, V[0, 2b]=-1
    # c=4: M5 = B11 -> V[1, 2b+1]=1
    # c=5: M6 = (B00+B01) -> V[0, 2b]=1, V[0, 2b+1]=1
    # c=6: M7 = (B10+B11) -> V[1, 2b]=1, V[1, 2b+1]=1
    
    # Strassen coefficients for W (C matrix indices k, i)
    # k in 0..11, i in 0..1
    # W[k*n + i, g]
    # For block b, k is 2b or 2b+1
    # c=0: M1 -> C00, C11 -> W[2b, 0]=1, W[2b+1, 1]=1
    # c=1: M2 -> C10, -C11 -> W[2b, 1]=1, W[2b+1, 1]=-1
    # c=2: M3 -> C01, C11 -> W[2b+1, 0]=1, W[2b+1, 1]=1
    # c=3: M4 -> C00, C10 -> W[2b, 0]=1, W[2b, 1]=1
    # c=4: M5 -> -C00, C01 -> W[2b, 0]=-1, W[2b+1, 0]=1
    # c=5: M6 -> C11 -> W[2b+1, 1]=1
    # c=6: M7 -> C00, C11 -> W[2b, 0]=1, W[2b+1, 1]=1

    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for c in range(rank_per_block):
            g = b * rank_per_block + c
            
            # Fill U
            for idx, val in U_coeffs[c].items():
                U[idx][g] = val
                
            # Fill V
            if c == 0:
                V[0 * p + k0][g] = 1
                V[1 * p + k1][g] = 1
            elif c == 1:
                V[0 * p + k0][g] = 1
            elif c == 2:
                V[0 * p + k1][g] = 1
                V[1 * p + k1][g] = -1
            elif c == 3:
                V[1 * p + k0][g] = 1
                V[0 * p + k0][g] = -1
            elif c == 4:
                V[1 * p + k1][g] = 1
            elif c == 5:
                V[0 * p + k0][g] = 1
                V[0 * p + k1][g] = 1
            elif c == 6:
                V[1 * p + k0][g] = 1
                V[1 * p + k1][g] = 1
                
            # Fill W
            if c == 0:
                W[k0 * n + 0][g] = 1
                W[k1 * n + 1][g] = 1
            elif c == 1:
                W[k0 * n + 1][g] = 1
                W[k1 * n + 1][g] = -1
            elif c == 2:
                W[k1 * n + 0][g] = 1
                W[k1 * n + 1][g] = 1
            elif c == 3:
                W[k0 * n + 0][g] = 1
                W[k0 * n + 1][g] = 1
            elif c == 4:
                W[k0 * n + 0][g] = -1
                W[k1 * n + 0][g] = 1
            elif c == 5:
                W[k1 * n + 1][g] = 1
            elif c == 6:
                W[k0 * n + 0][g] = 1
                W[k1 * n + 1][g] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
