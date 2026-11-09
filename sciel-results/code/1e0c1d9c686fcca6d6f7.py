import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct the schoolbook decomposition
    # U[i*m + j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p + k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n + i, r] = 1 if r corresponds to (i,j,k)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
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
                
    # Convert to lists for the interface
    def to_list(mat):
        res = []
        for row in mat:
            l = []
            for val in row:
                # Values are 0.0 or 1.0
                if val == 0.0:
                    l.append(0)
                else:
                    l.append(1)
            res.append(l)
        return res
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
