"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_fast(U, V, W):
    """
    Fast vectorized verification.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    # Reshape to 3D for einsum
    # U[i, j, r]
    U3 = U.reshape(n, m, r)
    # V[j, k, r]
    V3 = V.reshape(m, p, r)
    # W[k, i, r] -- Note: W is stored as (p*n, r) with index k*n + i
    # So W_flat[k*n + i, r] corresponds to W3[k, i, r]
    W3 = W.reshape(p, n, r)
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _try_random_rank(target_rank, timeout=5):
    """
    Attempt to find a random rank decomposition using fast verification.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < timeout:
        attempts += 1
        # Use integers {-1, 0, 1}
        # We can also try {-1, 1} for higher density if needed, but {-1,0,1} is standard
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        if _verify_fast(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Optional: Add a small chance to try half-integers?
        # For now, stick to integers for speed and simplicity.
        
    return None

def construct():
    deadline = time.time() + 110  # 10s margin for 120s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try to find lower rank
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    
    # We iterate from 25 upwards to find the minimal rank.
    # However, finding rank 25 might be hard. We can start from a higher rank and go down?
    # No, the goal is minimal. So we should start from the lower bound.
    # But if rank 25 is impossible, we waste time.
    # Given the search space is huge, it's better to try a range.
    # Let's try from 25 to 74.
    
    for r in range(25, best_rank):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Allocate time for each rank.
        # Ranks closer to the lower bound are harder? Or easier?
        # Usually, lower rank is harder to find randomly.
        # Let's give more time to lower ranks?
        # Or just a fixed small time per rank to cover many ranks.
        timeout = min(5.0, remaining / 2.0)
        
        res = _try_random_rank(r, timeout=timeout)
        if res:
            return res
        
    return best
# EVOLVE-BLOCK-END
