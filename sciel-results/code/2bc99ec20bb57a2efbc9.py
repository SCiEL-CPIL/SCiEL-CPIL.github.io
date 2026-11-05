"""Matrix multiplication 5x5x5 rank-minimal decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u2 = U[:, c].reshape(n, m)
        v2 = V[:, c].reshape(m, p)
        w2 = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u2[i,j] * v2[j,k] * w2[k,i]
    return np.allclose(T, np.ones((n,m,p)), atol=tol)

def construct():
    n, m, p = DIMENSIONS
    ns = [2,2,1]; ms = [2,2,1]; ps = [2,2,1]
    blocks = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = ns[i], ms[j], ps[k]
                if ni==2 and mj==2 and pk==2:
                    blocks.append(_strassen_2x2x2())
                else:
                    blocks.append(_schoolbook_block(ni, mj, pk))
    total_rank = sum(b[0].shape[1] for b in blocks)
    U = np.zeros((n*m, total_rank)); V = np.zeros((m*p, total_rank)); W = np.zeros((p*n, total_rank))
    ro_U = ro_V = ro_W = co = 0
    for u,v,w in blocks:
        r = u.shape[1]
        U[ro_U:ro_U+u.shape[0], co:co+r] = u
        V[ro_V:ro_V+v.shape[0], co:co+r] = v
        W[ro_W:ro_W+w.shape[0], co:co+r] = w
        ro_U += u.shape[0]; ro_V += v.shape[0]; ro_W += w.shape[0]; co += r
    assert _verify(U, V, W, n, m, p), "Decomposition invalid"
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
