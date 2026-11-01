import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook (rank 50)
    U, V, W = schoolbook_decomposition(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Try to reduce rank using gradient descent
    # Target rank 25
    for target_rank in [25, 24, 23, 22]:
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            # Random initialization
            np.random.seed(attempt)
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                # Compute error
                error = 0
                dU = np.zeros_like(U_rand)
                dV = np.zeros_like(V_rand)
                dW = np.zeros_like(W_rand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U_rand[idx_u, :], V_rand[idx_v, :] * W_rand[idx_w, :])
                            err = val - 1.0
                            error += err * err
                            
                            dU[idx_u, :] += 2 * err * (V_rand[idx_v, :] * W_rand[idx_w, :])
                            dV[idx_v, :] += 2 * err * (U_rand[idx_u, :] * W_rand[idx_w, :])
                            dW[idx_w, :] += 2 * err * (U_rand[idx_u, :] * V_rand[idx_v, :])
                
                if error < 1e-10:
                    break
                
                U_rand -= 0.01 * dU
                V_rand -= 0.01 * dV
                W_rand -= 0.01 * dW
                
            if error < 1e-8:
                # Snap to rationals
                for mult in [10, 100, 1000]:
                    U_snap = np.round(U_rand * mult) / mult
                    V_snap = np.round(V_rand * mult) / mult
                    W_snap = np.round(W_rand * mult) / mult
                    if validate_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        best_result = (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                        best_rank = target_rank
                        found = True
                        break
            if found:
                break
        if found:
            break
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
