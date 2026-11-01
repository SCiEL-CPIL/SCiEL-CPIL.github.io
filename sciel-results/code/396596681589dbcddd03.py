"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _verify_factors_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast vectorized verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
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

def _try_random_search(n, m, p, target_rank, trials=100000, seed=42, time_limit=30):
    """Random search for a low-rank decomposition with small integer entries."""
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        # Use small integer entries
        val_range = rng.choice([1, 2, 3])
        U = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_factors_fast(U, V, W, n, m, p):
            return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Try ranks from low to high
    for target_rank in [12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 15:
            break
        
        time_budget = min(remaining_time - 10, 20)
        
        result = _try_random_search(n, m, p, target_rank, trials=200000, seed=target_rank * 777, time_limit=time_budget)
        if result is not None:
            U, V, W, rank = result
            if _verify_factors_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
