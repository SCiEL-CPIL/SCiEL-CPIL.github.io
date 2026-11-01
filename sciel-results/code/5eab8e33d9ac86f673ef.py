"""Rewritten program: search for rank-10 exact tensor decomposition of 2x2 * 2x3 matmul."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _try_rank_systematic(n, m, p, r, max_attempts=100):
    """
    Try to find a rank-r decomposition using structured random search.
    For 2x2*2x3, n=2, m=2, p=3, r=10.
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    for attempt in range(max_attempts):
        rng = np.random.RandomState(attempt * 137 + 42)
        
        # Random initialization
        U = rng.randn(n_m, r)
        V = rng.randn(m_p, r)
        W = rng.randn(p_n, r)
        
        # Alternating least squares with damping
        for it in range(200):
            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        U_new[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        V_new[j*p+k, :] = x
                    except:
                        pass
            
            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        W_new[k*n+i, :] = x
                    except:
                        pass
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < 1e-12:
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    
    # Try rank 10
    if time.time() - start_time < deadline - 60:
        result = _try_rank_systematic(n, m, p, 10, max_attempts=30)
        if result is not None:
            U, V, W = result
            rank = 10
            U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 11
    if time.time() - start_time < deadline - 30:
        result = _try_rank_systematic(n, m, p, 11, max_attempts=20)
        if result is not None:
            U, V, W = result
            rank = 11
            U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 12
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
