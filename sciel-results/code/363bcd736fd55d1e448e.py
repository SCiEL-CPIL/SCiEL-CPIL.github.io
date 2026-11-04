import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce the rank of a decomposition by removing redundant columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return None
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = current_rank
    
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        if best_rank <= target_rank:
            break
        
        improved = False
        for c in range(best_rank):
            if time.time() > deadline - 5.0:
                break
            
            U_test = np.delete(best_U, c, axis=1)
            V_test = np.delete(best_V, c, axis=1)
            W_test = np.delete(best_W, c, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0
                        actual = 0.0
                        for cc in range(len(U_test[0])):
                            actual += U_test[i * m + j, cc] * V_test[j * p + k, cc] * W_test[k * n + i, cc]
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = len(U_test[0])
                improved = True
                break
        
        if not improved:
            break
    
    if best_rank < current_rank:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_random_decomposition(n, m, p, rank, seed, deadline, max_iters=500):
    """Try to find a rank-`rank` decomposition using random initialization and gradient-like refinement."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for iteration in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = val - 1.0
                    loss += err * err
                    for c in range(rank):
                        grad_U[i * m + j, c] += 2 * err * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k, c] += 2 * err * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i, c] += 2 * err * U[i * m + j, c] * V[j * p + k, c]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if best_loss < 1e-12:
            break
        
        lr *= 0.995
    
    if best_loss < 1e-9:
        U_list = [[round(float(x), 6) for x in row] for row in best_U]
        V_list = [[round(float(x), 6) for x in row] for row in best_V]
        W_list = [[round(float(x), 6) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    reduced = _try_reduce_rank(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p, 33, deadline)
    if reduced is not None:
        best_result = reduced
        best_rank = len(reduced["U"][0])
    
    min_rank = max(n * m, m * p, p * n)
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        for seed in range(5):
            if time.time() > deadline - 3.0:
                break
            
            result = _try_random_decomposition(n, m, p, rank, seed, deadline, max_iters=100)
            if result is not None:
                if rank < best_rank:
                    best_result = result
                    best_rank = rank
                break
    
    return best_result
