import time
import numpy as np

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 110

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 16
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # r maps to (j, k) where r = 4*j + k
    for j in range(m):
        for k in range(p):
            r = 4 * j + k
            
            # U: u = i*4 + j. Non-zero for all i in {0, 1}
            # U[i*4 + j, r] = 1
            for i in range(n):
                u_idx = i * m + j
                U[u_idx, r] = 1.0
                
            # V: v = j*4 + k. Non-zero only for matching j, k
            # V[j*4 + k, r] = 1
            v_idx = j * p + k
            V[v_idx, r] = 1.0
            
            # W: w = k*2 + i. Non-zero for all i in {0, 1}
            # W[k*2 + i, r] = 1
            for i in range(n):
                w_idx = k * n + i
                W[w_idx, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
