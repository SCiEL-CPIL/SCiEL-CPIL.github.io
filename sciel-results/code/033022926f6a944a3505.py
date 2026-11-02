import numpy as np

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Rank 56 decomposition using 8 independent 2x2x2 Strassen blocks
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            
            # A indices: A[i, 2a], A[i, 2a+1]
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            
            # B indices: B[2a+j, 2b+k]
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            
            # W indices: W[k*n + i]
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1
            
            # Strassen factors for 2x2x2
            # M1 = (A00+A11)(B00+B11)
            U[a00, base] = 1; U[a11, base] = 1
            V[b00, base] = 1; V[b11, base] = 1
            W[c00, base] = 1; W[c11, base] = 1
            
            # M2 = (A10+A11)B00
            U[a10, base+1] = 1; U[a11, base+1] = 1
            V[b00, base+1] = 1
            W[c10, base+1] = 1; W[c11, base+1] = -1
            
            # M3 = A00(B01-B11)
            U[a00, base+2] = 1
            V[b01, base+2] = 1; V[b11, base+2] = -1
            W[c01, base+2] = 1; W[c11, base+2] = 1
            
            # M4 = A11(B10-B00)
            U[a11, base+3] = 1
            V[b10, base+3] = 1; V[b00, base+3] = -1
            W[c00, base+3] = 1; W[c10, base+3] = 1
            
            # M5 = (A00+A01)B11
            U[a00, base+4] = 1; U[a01, base+4] = 1
            V[b11, base+4] = 1
            W[c00, base+4] = -1; W[c01, base+4] = 1
            
            # M6 = (A10-A00)(B00+B01)
            U[a10, base+5] = 1; U[a00, base+5] = -1
            V[b00, base+5] = 1; V[b01, base+5] = 1
            W[c11, base+5] = 1
            
            # M7 = (A01-A11)(B10+B11)
            U[a01, base+6] = 1; U[a11, base+6] = -1
            V[b10, base+6] = 1; V[b11, base+6] = 1
            W[c00, base+6] = 1
            
    # Convert to list for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
