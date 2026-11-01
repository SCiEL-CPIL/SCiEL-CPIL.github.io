<<<<<<< SEARCH
import numpy as np
U = np.zeros((4, 24))  # n*m = 4
V = np.zeros((22, 24))  # m*p = 22
W = np.zeros((22, 24))  # p*n = 22

# Term 0: A[0,0] * B[0,k] for all k
U[0, 0] = 1  # i=0, j=0
for k in range(11):
    V[0*11 + k, 0] = 1  # j=0, k
    W[k*2 + 0, 0] = 1  # k, i=0

# Term 1: A[1,1] * B[1,k] for all k
U[1*2 + 1, 1] = 1  # i=1, j=1
for k in range(11):
    V[1*11 + k, 1] = 1  # j=1, k
    W[k*2 + 1, 1] = 1  # k, i=1

# Terms 2..12: A[0,1] * B[1,k] for each k
for k in range(11):
    r = 2 + k
    U[0*2 + 1, r] = 1  # i=0, j=1
    V[1*11 + k, r] = 1  # j=1, k
    W[k*2 + 0, r] = 1  # k, i=0

# Terms 13..23: A[1,0] * B[0,k] for each k
for k in range(11):
    r = 13 + k
    U[1*2 + 0, r] = 1  # i=1, j=0
    V[0*11 + k, r] = 1  # j=0, k
    W[k*2 + 1, r] = 1  # k, i=1
=======
import numpy as np

def construct():
    n, m, p = 2, 2, 11
    rank = n * m * p  # Trivial rank 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
