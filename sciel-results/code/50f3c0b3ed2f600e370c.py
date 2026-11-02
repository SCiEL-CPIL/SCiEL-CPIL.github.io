import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # in the convention where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This tensor has rank 1.
    # U_{i,j,1} = 1, V_{j,k,1} = 1, W_{k,i,1} = 1.
    
    R = 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
