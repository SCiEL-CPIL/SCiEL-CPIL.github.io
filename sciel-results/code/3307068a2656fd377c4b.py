"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # Rows of U correspond to i*m + j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Strassen coefficients for B blocks (2x2)
    # Rows of V correspond to j*p + k
    # j=0, k=2u   -> 2u
    # j=0, k=2u+1 -> 2u+1
    # j=1, k=2u   -> 8+2u
    # j=1, k=2u+1 -> 8+2u+1
    
    # Output C reconstruction
    # Rows of W correspond to k*n + i
    # k=2u, i=0 -> 4u
    # k=2u, i=1 -> 4u+1
    # k=2u+1, i=0 -> 4u+2
    # k=2u+1, i=1 -> 4u+3
    
    for u in range(4):
        base_rank = u * 7
        # s=0: M1 = A00 * (B00 + B11)
        r = base_rank + 0
        U[0][r] = 1
        V[2*u][r] = 1
        V[8 + 2*u + 1][r] = 1
        W[4*u][r] = 1      # C00
        W[4*u + 3][r] = 1  # C11
        
        # s=1: M2 = (A00 + A01) * B10
        r = base_rank + 1
        U[0][r] = 1
        U[1][r] = 1
        V[8 + 2*u][r] = 1
        W[4*u + 1][r] = 1  # C10
        W[4*u + 3][r] = -1 # C11
        
        # s=2: M3 = (A10 + A11) * B01
        r = base_rank + 2
        U[2][r] = 1
        U[3][r] = 1
        V[2*u + 1][r] = 1
        W[4*u + 2][r] = 1  # C01
        
        # s=3: M4 = A11 * (B00 - B11)
        r = base_rank + 3
        U[3][r] = 1
        V[2*u][r] = 1
        V[8 + 2*u + 1][r] = -1
        W[4*u][r] = 1      # C00
        W[4*u + 1][r] = 1  # C10
        W[4*u + 3][r] = 1  # C11
        
        # s=4: M5 = (A00 + A10) * B11
        r = base_rank + 4
        U[0][r] = 1
        U[2][r] = 1
        V[8 + 2*u + 1][r] = 1
        W[4*u][r] = -1     # C00
        W[4*u + 2][r] = 1  # C01
        
        # s=5: M6 = (A01 - A10) * (B00 + B01)
        r = base_rank + 5
        U[1][r] = 1
        U[2][r] = -1
        V[2*u][r] = 1
        V[2*u + 1][r] = 1
        W[4*u + 3][r] = 1  # C11
        
        # s=6: M7 = (A11 - A01) * (B10 + B11)
        r = base_rank + 6
        U[3][r] = 1
        U[1][r] = -1
        V[8 + 2*u][r] = 1
        V[8 + 2*u + 1][r] = 1
        W[4*u][r] = 1      # C00
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
