import numpy as np
import time

def strassen_222():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def decompose(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return strassen_222()
    if n == 1 or m == 1 or p == 1:
        return schoolbook(n, m, p)
        
    if n == 2:
        # Split m or p
        if m >= 2 and p >= 2:
            # Split m into 2 and m-2? Or 1 and m-1?
            # To get 2x2x2, we need m>=2 and p>=2.
            # Split m into 1 and m-1?
            # If we split m into 1 and m-1, we get 2x1xp and 2x(m-1)xp.
            # 2x1xp is schoolbook.
            # We want to extract 2x2x2.
            # So split m into 2 and m-2.
            if m > 2:
                m1, m2 = 2, m-2
            else:
                m1, m2 = 1, 1
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            return combine(n, m1, m2, p, U1, V1, W1, U2, V2, W2, 'm')
        else:
            # m < 2 or p < 2 handled by schoolbook base case?
            # If m=1, schoolbook.
            return schoolbook(n, m, p)
            
    if n > 2:
        # Split n into 2 and n-2
        n1, n2 = 2, n-2
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        return combine(n1, m, n2, p, U1, V1, W1, U2, V2, W2, 'n')
        
    return schoolbook(n, m, p)

def combine(n1, m1, n2, p, U1, V1, W1, U2, V2, W2, axis):
    # This is tricky because the dimensions change.
    # It's easier to just concatenate and permute indices.
    # I will rewrite the combination logic in decompose directly.
    pass

# I will rewrite decompose to handle combination inline.
