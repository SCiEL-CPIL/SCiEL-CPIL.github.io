import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # Strassen rank-7 for 2x2 matrix multiplication
    strassen_terms = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]])),
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([[0,1],[0,1]])),
        (np.array([[0,0],[0,1]]), np.array([[-1,1],[0,0]]), np.array([[1,0],[1,0]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,1],[0,0]])),
        (np.array([[0,-1],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        (np.array([[0,1],[0,-1]]), np.array([[0,1],[1,0]]), np.array([[1,0],[0,0]])),
    ]
    
    rank = 28
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    # C1 = A1*B11 + A2*B21
    for u, v, w in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i*m + j, col] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, col] = v[j, k]
        for i in range(2):
            for k in range(2):
                W[k*n + i, col] = w[i, k]
        col += 1
    
    for u, v, w in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), col] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[(j+2)*p + k, col] = v[j, k]
        for i in range(2):
            for k in range(2):
                W[k*n + i, col] = w[i, k]
        col += 1
    
    # C2 = A1*B12 + A2*B22
    for u, v, w in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i*m + j, col] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[j*p + (k+2), col] = v[j, k]
        for i in range(2):
            for k in range(2):
                W[(k+2)*n + i, col] = w[i, k]
        col += 1
    
    for u, v, w in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), col] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[(j+2)*p + (k+2), col] = v[j, k]
        for i in range(2):
            for k in range(2):
                W[(k+2)*n + i, col] = w[i, k]
        col += 1
    
    assert col == rank
    
    # Verify
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_recon = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n).T
        ua = np.sum(u_r * A_test)
        vb = np.sum(v_r * B_test)
        C_recon += w_r * ua * vb
    
    assert np.allclose(C_recon, C_expected, atol=1e-10)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
