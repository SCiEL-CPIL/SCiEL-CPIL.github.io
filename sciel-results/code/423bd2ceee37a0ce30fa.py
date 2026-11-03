DIMENSIONS = [4, 5, 6]
import time
import numpy as np

def _strassen_222():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _trivial_221():
    n, m, p = 2, 2, 1
    rank = 2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    U[1, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    return U, V, W

def _embed_223():
    n, m, p = 2, 2, 3
    U222, V222, W222 = _strassen_222()
    U221, V221, W221 = _trivial_221()
    
    V222_padded = np.zeros((m*p, 7))
    V222_padded[:4, :] = V222
    
    U = np.hstack([U222, U221])
    V = np.hstack([V222_padded, V221])
    W = np.hstack([W222, W221])
    return U, V, W

def _trivial_212():
    n, m, p = 2, 1, 2
    rank = 2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1
    U[0, 1] = 1; U[2, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    return U, V, W

def _trivial_231():
    n, m, p = 2, 3, 1
    rank = 3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for j in range(3):
        U[0, j] = 1; U[2, j] = 1
        V[j, j] = 1
        W[0, j] = 1; W[1, j] = 1
    return U, V, W

def _embed_232():
    n, m, p = 2, 3, 2
    U222, V222, W222 = _strassen_222()
    U212, V212, W212 = _trivial_212()
    
    V222_padded = np.zeros((m*p, 7))
    V222_padded[:4, :] = V222
    V212_padded = np.zeros((m*p, 2))
    V212_padded[:2, :] = V212
    
    U = np.hstack([U222, U212])
    V = np.hstack([V222_padded, V212_padded])
    W = np.hstack([W222, W212])
    return U, V, W

def _embed_233():
    n, m, p = 2, 3, 3
    U232, V232, W232 = _embed_232()
    U231, V231, W231 = _trivial_231()
    
    W232_padded = np.zeros((p*n, 9))
    W232_padded[:4, :] = W232
    W231_padded = np.zeros((p*n, 3))
    W231_padded[:2, :] = W231
    
    U = np.hstack([U232, U231])
    V = np.hstack([V232, V231])
    W = np.hstack([W232_padded, W231_padded])
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 5, 6
    
    U223, V223, W223 = _embed_223()
    U233, V233, W233 = _embed_233()
    
    if time.time() > deadline:
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    U253 = np.hstack([U233, U223])
    V253 = np.hstack([V233, V223])
    W253 = np.hstack([W233, W223])
    
    U256 = np.hstack([U253, U253])
    V253_pad1 = np.zeros((m*p, 21))
    V253_pad1[:15, :] = V253
    V253_pad2 = np.zeros((m*p, 21))
    V253_pad2[15:30, :] = V253
    V256 = np.hstack([V253_pad1, V253_pad2])
    W253_pad1 = np.zeros((p*n, 21))
    W253_pad1[:6, :] = W253
    W253_pad2 = np.zeros((p*n, 21))
    W253_pad2[6:12, :] = W253
    W256 = np.hstack([W253_pad1, W253_pad2])
    
    U256_pad1 = np.zeros((n*m, 42))
    U256_pad1[:10, :] = U256
    U256_pad2 = np.zeros((n*m, 42))
    U256_pad2[10:20, :] = U256
    U456 = np.hstack([U256_pad1, U256_pad2])
    V456 = np.hstack([V256, V256])
    W456 = np.hstack([W256, W256])
    
    return {"U": U456.tolist(), "V": V456.tolist(), "W": W456.tolist()}
