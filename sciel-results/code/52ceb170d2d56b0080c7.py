import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The interface requires 3 matrices U, V, W of shapes (n*m, R), (m*p, R), (p*n, R).
    # This implies a decomposition of a 3rd order tensor T_{(i,j), (j,k), (k,i)}.
    # The "trace convention" W[k*n+i, r] and the cyclic indices i->j->k->i 
    # strongly suggest the tensor for trace(ABC) = sum A_ij B_jk C_ki.
    # The tensor for trace(ABC) is T_{i,j,k} = 1 for all i,j,k.
    # The rank of the all-ones tensor is 1.
    # U, V, W are vectors of ones.
    
    rank = 1
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = [[1] for _ in range(size_u)]
    V = [[1] for _ in range(size_v)]
    W = [[1] for _ in range(size_w)]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
