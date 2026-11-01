import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    r = 0
    # 5 Strassen blocks
    for block in range(5):
        b0, b1 = 2*block, 2*block+1
        c0, c1 = 2*block, 2*block+1
        
        # M1: (A00+A11)*(B00+B11)
        U[0, r] += 1; U[3, r] += 1
        V[b0, r] += 1; V[p+b1, r] += 1
        W[c0*2, r] += 1; W[c1*2+1, r] += 1
        r += 1
        
        # M2: (A10+A11)*B00
        U[2, r] += 1; U[3, r] += 1
        V[b0, r] += 1
        W[c0*2+1, r] += 1; W[c1*2+1, r] -= 1
        r += 1
        
        # M3: A00*(B01-B11)
        U[0, r] += 1
        V[b1, r] += 1; V[p+b1, r] -= 1
        W[c1*2, r] += 1; W[c1*2+1, r] += 1
        r += 1
        
        # M4: A11*(B10-B00)
        U[3, r] += 1
        V[p+b0, r] += 1; V[b0, r] -= 1
        W[c0*2, r] += 1; W[c0*2+1, r] += 1
        r += 1
        
        # M5: (A00+A01)*B11
        U[0, r] += 1; U[1, r] += 1
        V[p+b1, r] += 1
        W[c0*2, r] -= 1; W[c1*2, r] += 1
        r += 1
        
        # M6: (A10-A00)*(B00+B01)
        U[2, r] += 1; U[0, r] -= 1
        V[b0, r] += 1; V[b1, r] += 1
        W[c1*2+1, r] += 1
        r += 1
        
        # M7: (A01-A11)*(B10+B11)
        U[1, r] += 1; U[3, r] -= 1
        V[p+b0, r] += 1; V[p+b1, r] += 1
        W[c0*2, r] += 1
        r += 1

    # 1 Trivial block (col 10)
    b0 = 10
    c0 = 10
    # A00*B0
    U[0, r] = 1; V[b0, r] = 1; W[c0*2, r] = 1
    r += 1
    # A01*B1
    U[1, r] = 1; V[p+b0, r] = 1; W[c0*2, r] = 1
    r += 1
    # A10*B0
    U[2, r] = 1; V[b0, r] = 1; W[c0*2+1, r] = 1
    r += 1
    # A11*B1
    U[3, r] = 1; V[p+b0, r] = 1; W[c0*2+1, r] = 1
    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
