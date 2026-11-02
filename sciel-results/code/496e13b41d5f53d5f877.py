import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The tensor M_{n,m,p} is the sum over i,j,k of e_i^A e_j^B e_k^C e_i^C? 
    # No, it's sum over i,j,k of A_{ij} B_{jk} C_{ik}.
    # For each (i, j, k), we have a rank-1 term.
    # U_{i*m+j, c} = 1
    # V_{j*p+k, c} = 1
    # W_{k*n+i, c} = 1
    # where c is the index for (i, j, k).
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
                c += 1
                
    # Convert to lists of lists to match the return type if required, 
    # though the prompt says "integer/half-integer entries", numpy arrays are usually accepted 
    # or we can return lists. The seed returns lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
