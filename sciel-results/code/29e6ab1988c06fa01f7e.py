import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_matmul_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    
    # Check if T matches the expected tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1.0:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
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
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank by searching for better decompositions
    for target_rank in range(30, 42):  # Try ranks from 30 to 41
        if time.time() > deadline:
            break
            
        # Try random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            U_rand = [[int(np.random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[int(np.random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[int(np.random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = [row[:] for row in U_rand], [row[:] for row in V_rand], [row[:] for row in W_rand]
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
