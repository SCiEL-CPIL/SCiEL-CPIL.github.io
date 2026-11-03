import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    
    # Construct Schoolbook Decomposition (Rank 63)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1.0
                V_school[j*p + k, idx] = 1.0
                W_school[k*n + i, idx] = 1.0
                idx += 1
                
    return {'U': U_school, 'V': V_school, 'W': W_school}
