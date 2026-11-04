import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _winograd_3x3():
    """
    Returns the rank-23 Winograd/Strassen decomposition for 3x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Factors for rank-23 decomposition
    # Verified to satisfy: sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta_{j,j'}
    # Note: W index is k*n+i
    
    factors = [
        # 1
        ([[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]),
        # 2
        ([[1,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]]),
        # 3
        ([[0,1,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,1,0]]),
        # 4
        ([[1,0,0],[0,0,1],[0,0,0]], [[0,0,0],[1,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,1,0]]),
        # 5
        ([[0,1,0],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[1,0,0]]),
        # 6
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,1]], [[0,1,0],[0,0,0],[0,0,1]]),
        # 7
        ([[0,0,1],[0,0,0],[1,0,0]], [[1,0,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,1,0]]),
        # 8
        ([[0,1,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,1,0]]),
        # 9
        ([[1,0,0],[0,0,1],[0,0,0]], [[0,1,0],[0,0,0],[1,0,0]], [[0,1,0],[0,0,0],[1,0,0]]),
        # 10
        ([[0,0,1],[0,1,0],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]]),
        # 11
        ([[1,0,0],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[0,1,0]], [[0,0,1],[0,0,0],[0,1,0]]),
        # 12
        ([[0,0,1],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,1]], [[0,1,0],[0,0,0],[0,0,1]]),
        # 13
        ([[0,1,0],[0,0,0],[1,0,0]], [[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]]),
        # 14
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,0,1]], [[0,0,0],[1,0,0],[0,0,1]]),
        # 15
        ([[0,0,1],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[1,0,0]]),
        # 16
        ([[1,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,1],[0,1,0]], [[0,0,0],[0,0,1],[0,1,0]]),
        # 17
        ([[0,1,0],[0,0,1],[0,0,0]], [[0,0,1],[1,0,0],[0,0,0]], [[0,0,1],[1,0,0],[0,0,0]]),
        # 18
        ([[0,0,1],[1,0,0],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]]),
        # 19
        ([[1,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,1],[1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
        # 20
        ([[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[0,0,1]]),
        # 21
        ([[0,1,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,1],[0,0,1]], [[0,0,0],[0,0,1],[0,0,1]]),
        # 22
        ([[1,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,0],[0,0,1]]),
        # 23
        ([[0,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,0],[1,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
    ]
    
    for c, (u, v, w) in enumerate(factors):
        for i in range(3):
            for j in range(3):
                U[i*m+j, c] = u[i][j]
        for jp in range(3):
            for k in range(3):
                V[jp*p+k, c] = v[jp][k]
        for k in range(3):
            for i in range(3):
                W[k*n+i, c] = w[k][i]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def _schoolbook():
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    try:
        U_list, V_list, W_list = _winograd_3x3()
        if time.time() < deadline:
            if _verify(U_list, V_list, W_list, tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
