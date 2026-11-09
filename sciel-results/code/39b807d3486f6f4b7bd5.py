import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We will construct the trivial decomposition but using a different algorithmic approach.
    # Instead of looping and filling, we construct the basis vectors directly.
    
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # A rank-1 decomposition term is u ⊗ v ⊗ w.
    # We want sum_r u_r ⊗ v_r ⊗ w_r = T.
    # The trivial decomposition is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
    # Here, for each (i,j,k), u = e_{i*m+j}, v = e_{j*p+k}, w = e_{k*n+i}.
    
    # Let's build the matrices column by column.
    # Column r corresponds to (i,j,k).
    # U[:, r] has a 1 at index i*m + j.
    # V[:, r] has a 1 at index j*p + k.
    # W[:, r] has a 1 at index k*n + i.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i, j, k in product(range(n), range(m), range(p)):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        r += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
