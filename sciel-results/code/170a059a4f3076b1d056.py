import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Trivial construction: rank = n*m*p
    # For each (i, j, k), create a rank-1 term
    # U[i*m + j, r] = 1, V[j*p + k, r] = 1, W[k*n + i, r] = 1
    # where r is the index for the triple (i, j, k)
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
