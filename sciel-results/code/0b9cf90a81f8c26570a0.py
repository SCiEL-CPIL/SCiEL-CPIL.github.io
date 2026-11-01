"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Products r=0..6
    # U indices: i*m+k
    # V indices: k*p+j
    # W indices: j*n+i
    
    for b in range(3):
        # j offset for this block
        j_off = 2 * b
        
        # r = 0
        # M1 = (A00+A11)(B00+B11)
        U[0*2+0][b*7+0] = 1
        U[1*2+1][b*7+0] = 1
        V[0*6+j_off][b*7+0] = 1
        V[1*6+(j_off+1)][b*7+0] = 1
        W[(j_off)*2+0][b*7+0] = 1
        W[(j_off+1)*2+1][b*7+0] = 1
        
        # r = 1
        # M2 = (A10+A11)B00
        U[1*2+0][b*7+1] = 1
        U[1*2+1][b*7+1] = 1
        V[0*6+j_off][b*7+1] = 1
        W[(j_off)*2+1][b*7+1] = 1
        W[(j_off+1)*2+1][b*7+1] = -1
        
        # r = 2
        # M3 = A00(B10-B11)
        U[0*2+0][b*7+2] = 1
        V[1*6+j_off][b*7+2] = 1
        V[1*6+(j_off+1)][b*7+2] = -1
        W[(j_off+1)*2+0][b*7+2] = 1
        W[(j_off+1)*2+1][b*7+2] = 1
        
        # r = 3
        # M4 = A11(B01-B00)
        U[1*2+1][b*7+3] = 1
        V[0*6+(j_off+1)][b*7+3] = 1
        V[0*6+j_off][b*7+3] = -1
        W[(j_off)*2+0][b*7+3] = 1
        W[(j_off)*2+1][b*7+3] = 1
        
        # r = 4
        # M5 = (A00+A10)B11
        U[0*2+0][b*7+4] = 1
        U[0*2+1][b*7+4] = 1
        V[1*6+(j_off+1)][b*7+4] = 1
        W[(j_off)*2+0][b*7+4] = -1
        W[(j_off+1)*2+0][b*7+4] = 1
        
        # r = 5
        # M6 = (A10-A00)(B00+B01)
        U[1*2+0][b*7+5] = 1
        U[0*2+0][b*7+5] = -1
        V[0*6+j_off][b*7+5] = 1
        V[0*6+(j_off+1)][b*7+5] = 1
        W[(j_off+1)*2+1][b*7+5] = 1
        
        # r = 6
        # M7 = (A01-A11)(B10+B11)
        U[0*2+1][b*7+6] = 1
        U[1*2+1][b*7+6] = -1
        V[1*6+j_off][b*7+6] = 1
        V[1*6+(j_off+1)][b*7+6] = 1
        W[(j_off)*2+0][b*7+6] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
