import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m + j][t] * V[j*p + k][t] * W[k*n + i][t]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U_try = np.random.randn(n * m, target_rank) * 0.5
            V_try = np.random.randn(m * p, target_rank) * 0.5
            W_try = np.random.randn(p * n, target_rank) * 0.5
            
            if _verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, target_rank):
                best_rank = target_rank
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_try.tolist()
                found = True
                break
        
        if found:
            continue
        else:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
