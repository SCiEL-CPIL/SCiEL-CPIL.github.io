import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen U forms (shared)
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    # U indices: i*m + k
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    U[0, 0] += 1; U[3, 0] += 1 # S1
    U[2, 1] += 1; U[3, 1] += 1 # S2
    U[0, 2] += 1               # S3
    U[3, 3] += 1               # S4
    U[0, 4] += 1; U[1, 4] += 1 # S5
    U[2, 5] += 1; U[0, 5] -= 1 # S6
    U[1, 6] += 1; U[3, 6] -= 1 # S7
    
    def add_strassen_block(col_start):
        """
        Add V and W contributions for Strassen block at columns col_start, col_start+1.
        """
        c0 = col_start
        c1 = col_start + 1
        
        # V indices: k*p + l
        # W indices: l*n + i
        
        # M1 = S1 * (B0,c0 + B1,c1)
        # C0,c0 += M1, C1,c1 += M1
        V[0*7 + c0, 0] += 1
        V[1*7 + c1, 0] += 1
        W[c0*2 + 0, 0] += 1
        W[c1*2 + 1, 0] += 1
        
        # M2 = S2 * B0,c0
        # C1,c0 += M2, C1,c1 -= M2
        V[0*7 + c0, 1] += 1
        W[c0*2 + 1, 1] += 1
        W[c1*2 + 1, 1] -= 1
        
        # M3 = S3 * (B0,c1 - B1,c1)
        # C0,c1 += M3, C1,c1 += M3
        V[0*7 + c1, 2] += 1
        V[1*7 + c1, 2] -= 1
        W[c1*2 + 0, 2] += 1
        W[c1*2 + 1, 2] += 1
        
        # M4 = S4 * (B1,c0 - B0,c0)
        # C0,c0 += M4, C1,c0 += M4
        V[1*7 + c0, 3] += 1
        V[0*7 + c0, 3] -= 1
        W[c0*2 + 0, 3] += 1
        W[c0*2 + 1, 3] += 1
        
        # M5 = S5 * B1,c1
        # C0,c0 -= M5, C0,c1 += M5
        V[1*7 + c1, 4] += 1
        W[c0*2 + 0, 4] -= 1
        W[c1*2 + 0, 4] += 1
        
        # M6 = S6 * (B0,c0 + B0,c1)
        # C1,c1 += M6
        V[0*7 + c0, 5] += 1
        V[0*7 + c1, 5] += 1
        W[c1*2 + 1, 5] += 1
        
        # M7 = S7 * (B1,c0 + B1,c1)
        # C0,c0 += M7
        V[1*7 + c0, 6] += 1
        V[1*7 + c1, 6] += 1
        W[c0*2 + 0, 6] += 1

    # Add blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Column 6
    c6 = 6
    
    # Contributions for S3 (r=2)
    # V[0, c6] += 1
    # W[c6, 0] += 1, W[c6, 1] += 1
    V[0*7 + c6, 2] += 1
    W[c6*2 + 0, 2] += 1
    W[c6*2 + 1, 2] += 1
    
    # Contributions for S4 (r=3)
    # V[1, c6] += 1
    # W[c6, 0] += 1, W[c6, 1] += 1
    V[1*7 + c6, 3] += 1
    W[c6*2 + 0, 3] += 1
    W[c6*2 + 1, 3] += 1
    
    # Contributions for S6 (r=5)
    # V[0, c6] += 1
    # W[c6, 1] += 1
    V[0*7 + c6, 5] += 1
    W[c6*2 + 1, 5] += 1
    
    # Contributions for S7 (r=6)
    # V[1, c6] += 1
    # W[c6, 0] += 1
    V[1*7 + c6, 6] += 1
    W[c6*2 + 0, 6] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
