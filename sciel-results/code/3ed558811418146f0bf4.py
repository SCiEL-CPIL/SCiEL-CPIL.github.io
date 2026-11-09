import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # Helper to create empty matrices
    def create_empty(shape):
        return np.zeros(shape, dtype=int)

    # Strassen 2x2x2 decomposition
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_2x2x2():
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # A11: U[0,0]=1, A22: U[3,0]=1
        # B11: V[0,0]=1, B22: V[3,0]=1
        # C11: W[0,0]=1, C12: W[1,0]=1, C21: W[2,0]=1, C22: W[3,0]=1
        # Wait, W index is k*2+i.
        # C11 (i=0, k=0): W[0,0]
        # C12 (i=0, k=1): W[2,0] ? No.
        # W index: k*n + i. n=2.
        # C[i, k].
        # i=0, k=0 -> W[0,0]
        # i=0, k=1 -> W[2,0]
        # i=1, k=0 -> W[1,0]
        # i=1, k=1 -> W[3,0]
        
        # M1 contributes to C11, C12, C21, C22?
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # We need to assign coefficients to W.
        
        # Term 1: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C11
        W[2, 0] = 1 # C12 (i=0, k=1)
        W[1, 0] = 1 # C21 (i=1, k=0)
        W[3, 0] = 1 # C22 (i=1, k=1)
        
        # Term 2: M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1 # C21
        
        # Term 3: M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1 # C12
        W[3, 2] = -1 # C22
        
        # Term 4: M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1 # C21
        W[0, 3] = -1 # C11
        
        # Term 5: M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C11
        W[2, 4] = 1 # C12
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C22
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C11
        W[3, 6] = -1 # C22
        
        return U, V, W

    # ... implementation of combination logic ...
