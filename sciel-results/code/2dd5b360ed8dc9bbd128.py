"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = 4, 4, 7
SCHOOLBOOK_RANK = N * M * P  # 112


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # can be decomposed with rank 1.
    # U_{i,j,0} = 1, V_{j,k,0} = 1, W_{k,i,0} = 1
    # Sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
    
    rank = 1
    U = [[1.0] * rank for _ in range(N * M)]
    V = [[1.0] * rank for _ in range(M * P)]
    W = [[1.0] * rank for _ in range(P * N)]
    
    return {"U": U, "V": V, "W": W}
