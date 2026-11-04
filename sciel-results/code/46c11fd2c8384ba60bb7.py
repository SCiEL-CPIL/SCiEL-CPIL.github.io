"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute decomposition
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                val = 0
                for c in range(rank):
                    val += U_arr[i, c] * V_arr[j, c] * W_arr[k, c]
                if T[i, j, k] != 0 and val != 1:
                    return False
                if T[i, j, k] == 0 and val != 0:
                    return False
    return True

def _construct_schoolbook():
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

def _random_search(max_rank, time_limit, seed=42):
    """Try random low-rank decompositions."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Try ranks from low to high
    for rank in range(20, max_rank + 1):
        if time.time() > time_limit:
            break
        for attempt in range(100):
            if time.time() > time_limit:
                break
            # Generate random entries: 0, 0.5, 1, -0.5, -1
            vals = [0, 0.5, 1, -0.5, -1]
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            # Verify
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check each tensor entry
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U_arr[idx_u], V_arr[idx_v])
                        val = np.dot(val, W_arr[idx_w])  # This is wrong, need element-wise
                        # Actually: sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
                        val = sum(U_arr[idx_u, c] * V_arr[idx_v, c] * W_arr[idx_w, c] for c in range(rank))
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also check that non-tensor entries are 0
            # This is automatically satisfied if we only care about the tensor entries
            # But we need to ensure the decomposition is exact
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _greedy_merge(start_rank, time_limit):
    """Try to merge terms from a higher-rank decomposition."""
    n, m, p = DIMENSIONS
    # Start from schoolbook and try to merge
    U = [[0] * start_rank for _ in range(n * m)]
    V = [[0] * start_rank for _ in range(m * p)]
    W = [[0] * start_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find pairs of columns that can be merged
    # This is complex, so skip for now
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try random search with increasing rank
    best_result = _construct_schoolbook()
    
    # Try ranks from 50 to 104
    for max_rank in [60, 70, 80, 90, 100, 104]:
        if time.time() > deadline:
            break
        result = _random_search(max_rank, deadline, seed=random.randint(0, 10000))
        if result is not None:
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
