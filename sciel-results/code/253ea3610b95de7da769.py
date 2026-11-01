import numpy as np

def _strassen_2x2_decomposition():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Indices: 0:00, 1:01, 2:10, 3:11
    # U, V, W rows correspond to these indices.
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1 # z10
    
    # M3
    U[0, 2] += 1
    V[3, 2] += 1; V[0, 2] -= 1
    W[2, 2] += 1; W[3, 2] += 1 # z01, z11
    
    # M4
    U[1, 3] += 1
    V[2, 3] += 1; V[3, 3] += 1
    W[0, 3] += 1; W[1, 3] += 1 # z00, z10
    
    # M5
    U[0, 4] += 1; U[1, 4] -= 1
    V[0, 4] += 1; V[2, 4] += 1
    W[0, 4] -= 1; W[2, 4] += 1 # z00, z01
    
    # M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[1, 5] += 1
    W[3, 5] += 1 # z11
    
    # M7
    U[3, 6] += 1; U[2, 6] -= 1
    V[0, 6] -= 1; V[1, 6] += 1
    W[0, 6] += 1 # z00
    
    # Verify
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # i,j,k in 0..1
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
                if s != 1:
                    return None, None, None
    return U, V, W
