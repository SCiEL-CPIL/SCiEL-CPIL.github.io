import numpy as np
from fractions import Fraction
import itertools, time

n, m, p = 3, 4, 6
# Build target tensor T[i,j,k] = 1
# Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
# Identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == T[i,j,k]
T = np.ones((n, m, p))

def verify(U, V, W):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    M = U @ V.T          # (n*m, m*p)
    C = M.reshape(n, m, m, p)
    out = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                out[i,j,k] = s
    return np.allclose(out, T, atol=1e-6)

# schoolbook
def schoolbook():
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return U, V, W

U,V,W = schoolbook()
print("schoolbook rank", U.shape[1], "verify", verify(U,V,W))
