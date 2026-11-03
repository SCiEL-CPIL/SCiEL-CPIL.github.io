import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """Return Strassen 2x2x2 decomposition as (U, V, W)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x5x6_strassen():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    StrU, StrV, StrW = _strassen_222()
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    for k_offset in [0, 2, 4]:
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = StrU[i * 2 + j, t]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            col += 1
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    for k_offset in [0, 2, 4]:
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + (2 + j), col] = StrU[i * 2 + j, t]
            for j in range(2):
                for k in range(2):
                    V[(2 + j) * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            col += 1
    
    # Block 3: A[:, 4:5] * B[4:5, :]
    for i in range(2):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_2x5x6_strassen()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
