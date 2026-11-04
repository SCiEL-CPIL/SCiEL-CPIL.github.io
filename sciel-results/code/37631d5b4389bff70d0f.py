import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can vectorize this.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We want to compute T[i,j,k] for all i,j,k.
    # This is a tensor contraction.
    # T = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R))
    # But U is (n*m, R), so U.reshape(n,m,R) is correct.
    T = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,R), V.reshape(m,p,R), W.reshape(p,n,R))
    return np.allclose(T, 1.0, atol=1e-9)

def search_rank(R, time_limit):
    n, m, p = DIMENSIONS
    start_time = time.time()
    while time.time() - start_time < time_limit:
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    return None

def construct():
    # Try to find rank 26
    result = search_rank(26, 1)
    if result:
        return result
    # Try to find rank 25
    result = search_rank(25, 1)
    if result:
        return result
    # Return schoolbook
    return schoolbook()
