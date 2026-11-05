"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import numpy as np
import time

N, M, P = 4, 4, 5


def construct():
    # Schoolbook decomposition: rank = N*M*P = 80
    # This is guaranteed correct
    total_rank = N * M * P
    U = np.zeros((N * M, total_rank))
    V = np.zeros((M * P, total_rank))
    W = np.zeros((P * N, total_rank))
    
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, idx] = 1.0
                V[j * P + k, idx] = 1.0
                W[k * N + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
