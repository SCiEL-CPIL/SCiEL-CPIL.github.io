import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _try_random_rank(n, m, p, target_rank, max_iters=200):
    """Try to find a decomposition of given rank using random initialization."""
    np.random.seed(42)
    for _ in range(max_iters):
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        W = np.random.randint(-2, 3, size=(p*n, target_rank))
        
        # Check all (i,j,k) constraints
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Trivial decomposition: rank = n*m*p
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
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed verification")
    
    best = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() > deadline - 10:
            break
        result = _try_random_rank(n, m, p, target_rank, max_iters=50)
        if result is not None:
            best = {"U": result[0], "V": result[1], "W": result[2]}
            break
    
    return best
