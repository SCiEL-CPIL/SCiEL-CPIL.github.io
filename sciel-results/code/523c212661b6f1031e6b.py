"""Tensor decomposition search for 3x4 * 4x8 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition: for random A,B, check that reconstructed C matches A@B."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The decomposition defines a bilinear map: 
    # C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k] / ... 
    # 
    # Actually, the correct interpretation: the trilinear form is
    # sum_{i,j,k,r} U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # where C = A@B.
    # 
    # So: sum_{i,j,k,r} U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k] should equal sum_{i,k} (A@B)[i,k]
    
    np.random.seed(12345)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        rhs = np.sum(C_expected)
        
        lhs = 0.0
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] * A[i,j] * B[j,k]
        
        if abs(lhs - rhs) > tol:
            return False
    
    return True

def _schoolbook(n, m, p):
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

def _gradient_descent(n, m, p, target_rank, num_iters=2000, lr=0.005, seed=42):
    """Gradient descent to find rank-target_rank decomposition."""
    np.random.seed(seed)
    random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(num_iters):
        if _time_left() < 5.0:
            break
        
        # Compute residual R[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
        R = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        R -= 1.0
        
        loss = np.sum(R**2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-14:
            break
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        dU[i*m+j, r] += 2 * R[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += 2 * R[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += 2 * R[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_loss < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None

def _try_rank(n, m, p, target_rank, num_seeds=10):
    """Try multiple random initializations for given target rank."""
    for seed in range(num_seeds):
        if _time_left() < 5.0:
            break
        result = _gradient_descent(n, m, p, target_rank, num_iters=1500, lr=0.005, seed=1000 + seed)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try ranks from low to high
    for target_rank in range(10, 80):
        if _time_left() < 15.0:
            break
        result = _try_rank(n, m, p, target_rank, num_seeds=8)
        if result is not None:
            best = result
            break
    
    return best
