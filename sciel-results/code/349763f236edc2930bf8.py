"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 6 blocks of 2 columns
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: (A00 + A11) * (B00 + B11) -> C00(+), C11(+)
        U[0*m+0][col] = 1; U[1*m+1][col] = 1
        V[0*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # M2: (A10 + A11) * B00 -> C10(+)
        U[1*m+0][col] = 1; U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1
        col += 1
        
        # M3: A00 * (B01 - B11) -> C01(+), C11(+)
        U[0*m+0][col] = 1
        V[0*p+k1][col] = 1; V[1*p+k1][col] = -1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # M4: A11 * (B10 - B00) -> C00(+), C10(+)
        U[1*m+1][col] = 1
        V[1*p+k0][col] = 1; V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1
        col += 1
        
        # M5: (A00 + A01) * B11 -> C00(-), C01(+)
        U[0*m+0][col] = 1; U[0*m+1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1
        col += 1
        
        # M6: (A10 - A00) * (B00 + B01) -> C11(+)
        U[1*m+0][col] = 1; U[0*m+0][col] = -1
        V[0*p+k0][col] = 1; V[0*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B10 + B11) -> C00(+), C11(+)
        U[0*m+1][col] = 1; U[1*m+1][col] = -1
        V[1*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1

    # Last column k=12
    k_last = 12
    
    # A00 * B0,12 -> C0,12
    U[0*m+0][col] = 1
    V[0*p+k_last][col] = 1
    W[k_last*n+0][col] = 1
    col += 1
    
    # A01 * B1,12 -> C0,12
    U[0*m+1][col] = 1
    V[1*p+k_last][col] = 1
    W[k_last*n+0][col] = 1
    col += 1
    
    # A10 * B0,12 -> C1,12
    U[1*m+0][col] = 1
    V[0*p+k_last][col] = 1
    W[k_last*n+1][col] = 1
    col += 1
    
    # A11 * B1,12 -> C1,12
    U[1*m+1][col] = 1
    V[1*p+k_last][col] = 1
    W[k_last*n+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
