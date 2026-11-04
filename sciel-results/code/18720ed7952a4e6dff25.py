import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def _check_time():
    return time.time() >= _DEADLINE

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(rank, max_iters=200, seed=0):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    for attempt in range(max_iters):
        if _check_time():
            return None
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with sparse random values
        for c in range(rank):
            for idx in range(n * m):
                if rng.random() < 0.5:
                    U[idx, c] = rng.choice([-1, 1])
            for idx in range(m * p):
                if rng.random() < 0.5:
                    V[idx, c] = rng.choice([-1, 1])
            for idx in range(p * n):
                if rng.random() < 0.5:
                    W[idx, c] = rng.choice([-1, 1])
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
            
        # Simple ALS refinement
        for _ in range(5):
            if _check_time():
                return None
            for c in range(rank):
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    found = False
                    for j in range(m):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if abs(u_val) > 1e-12 and abs(v_val) > 1e-12:
                            target = 1.0 / (u_val * v_val)
                            if not found:
                                W[idx, c] = target
                                found = True
                            elif abs(W[idx, c] - target) > 1e-9:
                                break
                    if not found:
                        W[idx, c] = 0.0
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
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
    best_result = (U, V, W)
    
    # Try to find lower rank
    for r in range(30, 63):
        if _check_time():
            break
        result = _try_rank(r, max_iters=50, seed=r)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            break
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
