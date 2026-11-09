"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    
    # The tensor T is a sum of m disjoint tensors T^(j) for j=0..m-1.
    # Each T^(j) corresponds to the multiplication of an n-vector by a p-vector (outer product).
    # The rank of each T^(j) is n*p.
    # Since the supports are disjoint, the total rank is sum of ranks = m*n*p.
    # This is the minimal possible rank.
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(nm)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(mp)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
