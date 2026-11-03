import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for _ in range(5):
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        C_calc = np.zeros((n, p))
        for r in range(rank):
            M = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    M[i] += A[i, j] * U[i*m+j, r]
            N = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    N[k] += B[j, k] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    C_calc[i, k] += W[k*n+i, r] * M[i] * N[k]
        if not np.allclose(C_calc, C_expected, atol=1e-5):
            return False
    return True

def find_decomposition(rank, time_limit=50):
    n, m, p = DIMENSIONS
    start_time = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook subset?
    # Just random start
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Simple gradient descent
    lr = 0.01
    while time.time() - start_time < time_limit:
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        C_calc = np.zeros((n, p))
        grads_U = np.zeros_like(U)
        grads_V = np.zeros_like(V)
        grads_W = np.zeros_like(W)
        
        M = np.zeros((n, rank))
        N = np.zeros((p, rank))
        
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    M[i, r] += A[i, j] * U[i*m+j, r]
                    
        for k in range(p):
            for j in range(m):
                for r in range(rank):
                    N[k, r] += B[j, k] * V[j*p+k, r]
                    
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    C_calc[i, k] += W[k*n+i, r] * M[i, r] * N[k, r]
                    
        err = np.sum((C_calc - C_expected)**2)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-6:
                break
                
        # Compute gradients
        # d(err)/d(W[k*n+i, r]) = 2 * (C_calc[i, k] - C_expected[i, k]) * M[i, r] * N[k, r]
        for k in range(p):
            for i in range(n):
                diff = 2 * (C_calc[i, k] - C_expected[i, k])
                for r in range(rank):
                    grads_W[k*n+i, r] = diff * M[i, r] * N[k, r]
                    
        # d(err)/d(U[i*m+j, r]) = sum_{k} 2 * (C_calc[i, k] - C_expected[i, k]) * W[k*n+i, r] * N[k, r] * A[i, j]
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    g = 0
                    for k in range(p):
                        g += 2 * (C_calc[i, k] - C_expected[i, k]) * W[k*n+i, r] * N[k, r]
                    grads_U[i*m+j, r] = g * A[i, j]
                    
        # d(err)/d(V[j*p+k, r]) = sum_{i} 2 * (C_calc[i, k] - C_expected[i, k]) * W[k*n+i, r] * M[i, r] * B[j, k]
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    g = 0
                    for i in range(n):
                        g += 2 * (C_calc[i, k] - C_expected[i, k]) * W[k*n+i, r] * M[i, r]
                    grads_V[j*p+k, r] = g * B[j, k]
                    
        U -= lr * grads_U
        V -= lr * grads_V
        W -= lr * grads_W
        
    # Round to integers
    if best_U is not None:
        U_int = np.round(best_U).astype(int)
        V_int = np.round(best_V).astype(int)
        W_int = np.round(best_W).astype(int)
        if verify_decomposition(U_int, V_int, W_int):
            return U_int, V_int, W_int
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    # Try 23
    res = find_decomposition(23, time_limit=100)
    if res:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
