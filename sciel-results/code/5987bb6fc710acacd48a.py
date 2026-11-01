import time
import sys

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # Each term is a list of (u_indices, v_indices, w_indices)
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k_offset, coeff) -- k is absolute, but we use offset in block
    # w_indices: list of (k_offset, i, coeff)
    
    # We will just hardcode the logic for the 7 terms.
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 1: M1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0*2+0][r] += 1
        U[1*2+1][r] += 1
        V[0*10+k0][r] += 1
        V[1*10+k1][r] += 1
        W[k0*2+0][r] += 1
        W[k1*2+1][r] += 1
        
        # Term 2: M2 = (c+d)e
        r = b * 7 + 1
        U[1*2+0][r] += 1
        U[1*2+1][r] += 1
        V[0*10+k0][r] += 1
        W[k0*2+1][r] += 1
        W[k1*2+1][r] += -1
        
        # Term 3: M3 = a(f-h)
        r = b * 7 + 2
        U[0*2+0][r] += 1
        V[0*10+k1][r] += 1
        V[1*10+k1][r] += -1
        W[k1*2+0][r] += 1
        W[k1*2+1][r] += 1
        
        # Term 4: M4 = d(g-e)
        r = b * 7 + 3
        U[1*2+1][r] += 1
        V[1*10+k0][r] += 1
        V[0*10+k0][r] += -1
        W[k0*2+0][r] += 1
        W[k0*2+1][r] += 1
        
        # Term 5: M5 = (a+b)h
        r = b * 7 + 4
        U[0*2+0][r] += 1
        U[0*2+1][r] += 1
        V[1*10+k1][r] += 1
        W[k0*2+0][r] += -1
        W[k1*2+0][r] += 1
        
        # Term 6: M6 = (c-a)(e+f)
        r = b * 7 + 5
        U[1*2+0][r] += 1
        U[0*2+0][r] += -1
        V[0*10+k0][r] += 1
        V[0*10+k1][r] += 1
        W[k1*2+1][r] += 1
        
        # Term 7: M7 = (b-d)(g+h)
        r = b * 7 + 6
        U[0*2+1][r] += 1
        U[1*2+1][r] += -1
        V[1*10+k0][r] += 1
        V[1*10+k1][r] += 1
        W[k0*2+0][r] += 1

    return {"U": U, "V": V, "W": W}
