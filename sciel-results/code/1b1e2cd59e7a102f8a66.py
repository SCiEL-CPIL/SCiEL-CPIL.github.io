import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 20
    
    U = np.zeros((n*m, rank))  # 8 x 20
    V = np.zeros((m*p, rank))  # 20 x 20
    W = np.zeros((p*n, rank))  # 10 x 20
    
    idx = 0
    for k in range(p):  # k = 0,1,2,3,4
        for j0 in [0, 2]:  # j-pairs: (0,1) and (2,3)
            # Term A: i=0, j=j0
            U[0*m + j0, idx] = 1
            V[j0*p + k, idx] = 1
            W[k*n + 0, idx] = 1
            idx += 1
            # Term B: i=1, j=j0+1
            U[1*m + (j0+1), idx] = 1
            V[(j0+1)*p + k, idx] = 1
            W[k*n + 1, idx] = 1
            idx += 1
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
